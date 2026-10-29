mod graph;
mod sources;

use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Cell, Paragraph, Row, Table};
use ratatui::Frame;
use sources::{Completed, LoadedModel, Msg, Source, SysSample};
use std::collections::{HashMap, VecDeque};
use std::sync::mpsc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Chart time windows in seconds; + zooms in (shorter), - zooms out (longer).
const WINDOWS: [f64; 7] = [15.0, 30.0, 60.0, 300.0, 900.0, 3600.0, 4.0 * 3600.0];
/// Index into WINDOWS used at startup (1 minute).
const DEFAULT_WINDOW: usize = 2;
/// Caps keep memory flat when left running for days.
const MAX_REQUESTS: usize = 5000;
const MAX_SAMPLES: usize = 4 * 3600;
/// RGB so rows can be faded (see `fade`).
const PALETTE: [Color; 8] = [
    Color::Rgb(0x56, 0xd4, 0xdd), Color::Rgb(0xc6, 0x78, 0xdd), Color::Rgb(0xe5, 0xc0, 0x7b), Color::Rgb(0x98, 0xc3, 0x79),
    Color::Rgb(0x61, 0xaf, 0xef), Color::Rgb(0xe0, 0x6c, 0x75), Color::Rgb(0xa6, 0xe2, 0x2e), Color::Rgb(0xff, 0x79, 0xc6),
];
const TEXT: Color = Color::Rgb(0xd0, 0xd0, 0xd0);
const BORDER: Color = Color::DarkGray;
/// System memory is shown in binary GB like macOS; model sizes stay decimal like LM Studio.
const GIB: f64 = (1u64 << 30) as f64;

/// Progress of a running llama.cpp request as (from, to, tok/s) intervals in unix seconds.
#[derive(Default)]
struct Live {
    input: Option<(f64, f64, f64)>,
    output: Vec<(f64, f64, f64)>,
    progress: f64,
    n_gen: u64,
    updated: f64,
}

impl Live {
    /// Seconds without a progress line before a request is treated as finished or aborted.
    const STALE: f64 = 4.0;

    fn active(&self, now: f64) -> bool {
        now - self.updated < Self::STALE
    }
}

struct App {
    done: VecDeque<Completed>,
    /// llama.cpp requests still running, keyed by (model, task).
    live: HashMap<(String, String), Live>,
    loaded: Vec<LoadedModel>,
    /// Remembered after unload so old requests keep their display name and format.
    names: HashMap<String, String>,
    formats: HashMap<String, String>,
    colors: HashMap<String, Color>,
    gpu: VecDeque<(f64, f64)>,
    mem: VecDeque<(f64, f64)>,
    sys: Option<SysSample>,
    win: usize,
}

fn unix_now() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs_f64()
}

fn unix(t: SystemTime) -> f64 {
    t.duration_since(UNIX_EPOCH).unwrap_or_default().as_secs_f64()
}

fn push_capped<T>(q: &mut VecDeque<T>, v: T, cap: usize) {
    if q.len() == cap {
        q.pop_front();
    }
    q.push_back(v);
}

impl App {
    fn new() -> Self {
        Self {
            done: VecDeque::new(), live: HashMap::new(), loaded: Vec::new(), names: HashMap::new(),
            formats: HashMap::new(), colors: HashMap::new(), gpu: VecDeque::new(), mem: VecDeque::new(),
            sys: None, win: DEFAULT_WINDOW,
        }
    }

    fn handle(&mut self, msg: Msg) {
        match msg {
            Msg::Completed(c) => {
                // GGUF models are covered exactly by llama.cpp timings; skip their duplicate stats event.
                if c.source == Source::Stats && self.formats.get(&c.model).is_some_and(|f| f == "gguf") {
                    return;
                }
                if let Some(task) = &c.task {
                    self.live.remove(&(c.model.clone(), task.clone()));
                }
                self.color(&c.model);
                push_capped(&mut self.done, c, MAX_REQUESTS);
            }
            Msg::Prefill { model, task, progress, tps, elapsed } => {
                let now = unix_now();
                let l = self.live.entry((model, task)).or_default();
                l.input = Some((now - elapsed, now, tps));
                l.progress = progress;
                l.updated = now;
            }
            Msg::Decode { model, task, n_gen, tps, tps_3s } => {
                let now = unix_now();
                let l = self.live.entry((model, task)).or_default();
                // The first line covers the whole generation so far at its average speed.
                let (from, rate) = match l.output.last() {
                    Some(last) => (last.1, tps_3s),
                    None => (now - n_gen as f64 / tps.max(f64::EPSILON), tps),
                };
                l.output.push((from, now, rate));
                l.progress = 1.0;
                l.n_gen = n_gen;
                l.updated = now;
            }
            Msg::Loaded(models) => {
                for m in &models {
                    self.names.insert(m.id.clone(), m.name.clone());
                    self.formats.insert(m.id.clone(), m.format.clone());
                    self.color(&m.id);
                }
                self.loaded = models;
            }
            Msg::Sys(s) => {
                let now = unix_now();
                self.live.retain(|_, l| now - l.updated < 30.0);
                push_capped(&mut self.gpu, (now, s.gpu), MAX_SAMPLES);
                if s.mem_total > 0 {
                    push_capped(&mut self.mem, (now, 100.0 * s.mem_used as f64 / s.mem_total as f64), MAX_SAMPLES);
                }
                self.sys = Some(s);
            }
        }
    }

    fn color(&mut self, model: &str) -> Color {
        let n = self.colors.len();
        *self.colors.entry(model.to_string()).or_insert(PALETTE[n % PALETTE.len()])
    }

    fn name(&self, model: &str) -> String {
        self.names.get(model).filter(|n| !n.is_empty()).cloned().unwrap_or_else(|| model.to_string())
    }
}

fn main() -> std::io::Result<()> {
    let (tx, rx) = mpsc::channel();
    let children = sources::Children::default();
    sources::start(tx, children.clone());

    let mut terminal = ratatui::init();
    let mut app = App::new();
    let result = (|| loop {
        while let Ok(msg) = rx.try_recv() {
            app.handle(msg);
        }
        terminal.draw(|f| draw(f, &app))?;
        if event::poll(Duration::from_millis(250))? {
            if let Event::Key(k) = event::read()? {
                if k.kind != KeyEventKind::Press {
                    continue;
                }
                match k.code {
                    KeyCode::Char('q') | KeyCode::Esc => return Ok(()),
                    KeyCode::Char('c') if k.modifiers.contains(KeyModifiers::CONTROL) => return Ok(()),
                    KeyCode::Char('+') | KeyCode::Char('=') => app.win = app.win.saturating_sub(1),
                    KeyCode::Char('-') | KeyCode::Char('_') => app.win = (app.win + 1).min(WINDOWS.len() - 1),
                    KeyCode::Char('0') => app.win = DEFAULT_WINDOW,
                    KeyCode::Char('c') => app.done.clear(),
                    _ => {}
                }
            }
        }
    })();
    ratatui::restore();
    for c in children.lock().unwrap().iter_mut() {
        let _ = c.kill();
    }
    result
}

fn panel(title: &str, color: Color) -> Block<'_> {
    Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(BORDER))
        .title(Line::from(vec![Span::raw(" "), Span::styled(title, Style::default().fg(color).bold()), Span::raw(" ")]))
}

/// Blends an RGB colour towards the dark background; `k` = 0 keeps it, 1 is fully faded.
fn fade(c: Color, k: f64) -> Color {
    let Color::Rgb(r, g, b) = c else { return c };
    let mix = |x: u8| (x as f64 + (0x2a as f64 - x as f64) * k) as u8;
    Color::Rgb(mix(r), mix(g), mix(b))
}

fn fmt_window(s: f64) -> String {
    match s {
        s if s >= 3600.0 => format!("{}h", s / 3600.0),
        s if s >= 60.0 => format!("{}m", s / 60.0),
        s => format!("{s}s"),
    }
}

fn fmt_tokens(n: u64) -> String {
    match n {
        n if n >= 1_000_000 => format!("{:.1}M", n as f64 / 1e6),
        n if n >= 10_000 => format!("{:.1}k", n as f64 / 1e3),
        n => n.to_string(),
    }
}

fn fmt_ago(secs: f64) -> String {
    match secs {
        s if s < 60.0 => format!("{s:.0}s ago"),
        s if s < 3600.0 => format!("{:.0}m ago", s / 60.0),
        s => format!("{:.1}h ago", s / 3600.0),
    }
}

fn draw(f: &mut Frame, app: &App) {
    let [header, sys, speed, in_use, bottom, footer] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(8),
        Constraint::Min(10),
        Constraint::Length(app.loaded.len().max(1) as u16 + 3),
        Constraint::Length(12),
        Constraint::Length(1),
    ])
    .areas(f.area());

    draw_header(f, header, app);
    let [gpu, mem] = Layout::horizontal([Constraint::Percentage(50); 2]).areas(sys);
    draw_sys(f, gpu, app, true);
    draw_sys(f, mem, app, false);
    let [out, inp] = Layout::horizontal([Constraint::Percentage(50); 2]).areas(speed);
    draw_speed(f, out, app, false);
    draw_speed(f, inp, app, true);
    draw_in_use(f, in_use, app);
    let [summary, recent] = Layout::horizontal([Constraint::Percentage(55), Constraint::Percentage(45)]).areas(bottom);
    draw_summary(f, summary, app);
    draw_recent(f, recent, app);

    let key = |k: &'static str| Span::styled(k, Style::default().fg(Color::Yellow).bold());
    f.render_widget(
        Paragraph::new(Line::from(vec![
            key(" q"), Span::raw(" quit  "),
            key("+/-"), Span::raw(" zoom  "), key("0"), Span::raw(format!(" reset zoom  (window {})  ", fmt_window(WINDOWS[app.win]))),
            key("c"), Span::raw(" clear  "),
            Span::styled("~ input speed estimated from time to first token (non-GGUF)", Style::default().fg(BORDER)),
        ])),
        footer,
    );
}

fn draw_header(f: &mut Frame, area: Rect, app: &App) {
    let now = chrono::Local::now().format("%H:%M:%S");
    let mut spans = vec![
        Span::styled(" lmstop ", Style::default().fg(Color::Black).bg(Color::Cyan).bold()),
        Span::raw(format!("  {now}  ")),
    ];
    if let Some(s) = &app.sys {
        spans.push(Span::styled(format!("GPU {:>3.0}%  ", s.gpu), Style::default().fg(Color::Green)));
        spans.push(Span::styled(
            format!("RAM {:.1}/{:.0} GB  ", s.mem_used as f64 / GIB, s.mem_total as f64 / GIB),
            Style::default().fg(Color::Magenta),
        ));
    }
    spans.push(Span::raw(format!("{} loaded  {} requests", app.loaded.len(), app.done.len())));
    f.render_widget(Paragraph::new(Line::from(spans)), area);
}

/// Column time slots are aligned to the wall clock (slot k covers [k*w, (k+1)*w)), so as time
/// passes the graph shifts by whole columns, like btop, instead of re-rasterising every frame.
struct Slots {
    w: f64,
    first: i64,
    cols: usize,
}

impl Slots {
    fn new(win: f64, cols: usize, now: f64) -> Self {
        let w = win / cols.max(1) as f64;
        Self { w, first: (now / w).floor() as i64 - cols as i64 + 1, cols }
    }

    fn col(&self, t: f64) -> Option<usize> {
        let c = (t / self.w).floor() as i64 - self.first;
        (0..self.cols as i64).contains(&c).then_some(c as usize)
    }

    fn start(&self, col: usize) -> f64 {
        (self.first + col as i64) as f64 * self.w
    }
}

/// Draws a btop-style braille graph: y labels on the left, one sample per dot column.
/// `series` gets the sample count and returns values (0..=1) plus the label of the top row.
fn graph(f: &mut Frame, area: Rect, title: &str, color: Color, fill: graph::Fill, win: f64, series: impl FnOnce(usize) -> (Vec<f64>, String)) {
    let block = panel(title, color)
        .title_bottom(Line::from(format!(" -{} ", fmt_window(win))).left_aligned().fg(BORDER))
        .title_bottom(Line::from(" now ").right_aligned().fg(BORDER));
    let inner = block.inner(area);
    f.render_widget(block, area);
    let [labels, plot] = Layout::horizontal([Constraint::Length(6), Constraint::Fill(1)]).areas(inner);
    let (values, top) = series(plot.width as usize * 2);
    let mut lines = vec![Line::from(""); labels.height as usize];
    if let Some(l) = lines.first_mut() { *l = Line::from(top); }
    if let Some(l) = lines.last_mut() { *l = Line::from("0"); }
    f.render_widget(Paragraph::new(lines).right_aligned().fg(BORDER), Rect { width: labels.width - 1, ..labels });
    f.render_widget(graph::Graph { values: &values, fill }, plot);
}

fn draw_sys(f: &mut Frame, area: Rect, app: &App, gpu: bool) {
    let win = WINDOWS[app.win];
    let now = unix_now();
    let (hist, color) = if gpu { (&app.gpu, Color::Green) } else { (&app.mem, Color::Magenta) };
    let title = match (&app.sys, gpu) {
        (Some(s), true) => format!("GPU {:.0}%  ·  GPU memory {:.1} GB", s.gpu, s.gpu_mem as f64 / GIB),
        (Some(s), false) => format!("Memory {:.1} / {:.0} GB", s.mem_used as f64 / GIB, s.mem_total as f64 / GIB),
        (None, true) => "GPU".into(),
        (None, false) => "Memory".into(),
    };
    graph(f, area, &title, color, graph::Fill::Gradient(&graph::HEAT), win, |cols| {
        let slots = Slots::new(win, cols, now);
        let mut vals: Vec<Option<f64>> = vec![None; cols];
        let mut before = None;
        for &(t, v) in hist {
            match slots.col(t) {
                Some(c) => vals[c] = Some(vals[c].map_or(v, |x: f64| x.max(v))),
                None if t < slots.start(0) => before = Some(v),
                None => {}
            }
        }
        // Samples arrive every second, so slots shorter than that hold the previous value.
        let mut last = before;
        let values = vals.iter().map(|v| {
            last = v.or(last);
            last.unwrap_or(0.0) / 100.0
        }).collect();
        (values, "100%".into())
    });
}

fn draw_speed(f: &mut Frame, area: Rect, app: &App, input: bool) {
    let win = WINDOWS[app.win];
    let now = unix_now();
    let last = app.done.iter().rev().find_map(|c| if input { c.prompt_tps } else { Some(c.gen_tps) });
    let (name, color) = if input { ("Input tok/s (prompt processing)", Color::LightBlue) } else { ("Output tok/s", Color::Cyan) };
    let title = match last {
        Some(v) => format!("{name}  ·  last {v:.1}"),
        None => name.to_string(),
    };
    graph(f, area, &title, color, graph::Fill::Solid(color), win, |cols| {
        let slots = Slots::new(win, cols, now);
        // (from, to, tok/s): each request spread over the time it actually spent in that phase.
        let mut spans: Vec<(f64, f64, f64)> = Vec::new();
        for c in &app.done {
            let end = unix(c.at);
            let gen_s = if c.gen_tps > 0.0 { c.gen_tokens as f64 / c.gen_tps } else { 0.0 };
            let prompt_s = c.ttft.unwrap_or(0.0);
            if !input {
                spans.push((end - gen_s, end, c.gen_tps));
            } else if let Some(tps) = c.prompt_tps {
                spans.push((end - gen_s - prompt_s, end - gen_s, tps));
            }
        }
        // Running requests; the latest speed is held until now so the graph moves between log lines.
        for l in app.live.values() {
            let fresh = l.active(now);
            if input {
                if let Some((from, to, tps)) = l.input {
                    spans.push((from, if fresh && l.output.is_empty() { now } else { to }, tps));
                }
            } else {
                spans.extend(&l.output);
                if let (true, Some(last)) = (fresh, l.output.last()) {
                    spans.push((last.1, now, last.2));
                }
            }
        }
        let mut total = vec![0.0f64; cols];
        for (from, to, rate) in spans {
            if to <= from || to < slots.start(0) {
                continue;
            }
            let (Some(a), b) = (slots.col(from.max(slots.start(0))), slots.col(to).unwrap_or(cols - 1)) else { continue };
            for col in a..=b {
                let s = slots.start(col);
                // The newest slot is still in progress: average over the part that has elapsed.
                let len = if col == cols - 1 { (now - s).clamp(1e-3, slots.w) } else { slots.w };
                let overlap = (to.min(s + slots.w) - from.max(s)).max(0.0) / len;
                total[col] += rate * overlap;
            }
        }
        let peak = total.iter().copied().fold(0.0, f64::max);
        let values = total.iter().map(|v| if peak > 0.0 { v / peak } else { 0.0 }).collect();
        (values, format!("{peak:.0}"))
    });
}

fn draw_in_use(f: &mut Frame, area: Rect, app: &App) {
    let now_ms = unix_now() * 1000.0;
    let header = Row::new(["", "Model", "Quant", "Size", "Context", "Slots", "Status", "Last used"])
        .style(Style::default().fg(Color::Yellow).bold());
    let rows: Vec<Row> = app.loaded.iter().map(|m| {
        let color = app.colors.get(&m.id).copied().unwrap_or(Color::White);
        let now = now_ms / 1000.0;
        let live = app.live.iter().find(|((model, _), l)| *model == m.id && l.active(now)).map(|(_, l)| l);
        let busy = m.status != "idle" || live.is_some();
        let status = match live {
            Some(l) if !l.output.is_empty() => Span::styled(
                format!("generating {:.1} tok/s · {}", l.output.last().map_or(0.0, |o| o.2), fmt_tokens(l.n_gen)),
                Style::default().fg(Color::Green).bold(),
            ),
            Some(l) => Span::styled(
                format!("prefill {:>3.0}% @ {:.0} tok/s", l.progress * 100.0, l.input.map_or(0.0, |i| i.2)),
                Style::default().fg(Color::LightBlue).bold(),
            ),
            None if busy => Span::styled(m.status.clone(), Style::default().fg(Color::Green).bold()),
            None => Span::styled(m.status.clone(), Style::default().fg(BORDER)),
        };
        let ago = (m.last_used > 0).then(|| fmt_ago((now_ms - m.last_used as f64).max(0.0) / 1000.0)).unwrap_or_default();
        Row::new(vec![
            Cell::from(Span::styled(if busy { "▶" } else { "●" }, Style::default().fg(color))),
            Cell::from(Span::styled(m.name.clone(), Style::default().fg(color).add_modifier(if busy { Modifier::BOLD } else { Modifier::empty() }))),
            Cell::from(m.quant.clone()),
            Cell::from(format!("{:.1} GB", m.size as f64 / 1e9)),
            Cell::from(fmt_tokens(m.ctx)),
            Cell::from(format!("{}{}", m.parallel, if m.queued > 0 { format!(" +{} queued", m.queued) } else { String::new() })),
            Cell::from(status),
            Cell::from(ago),
        ])
    }).collect();
    let title = if app.loaded.is_empty() { "In use  ·  no models loaded".to_string() } else { "In use".to_string() };
    let table = Table::new(rows, [
        Constraint::Length(1), Constraint::Fill(1), Constraint::Length(8), Constraint::Length(9),
        Constraint::Length(8), Constraint::Length(14), Constraint::Length(32), Constraint::Length(10),
    ])
    .header(header)
    .block(panel(&title, Color::Green));
    f.render_widget(table, area);
}

fn draw_summary(f: &mut Frame, area: Rect, app: &App) {
    #[derive(Default)]
    struct Agg { n: u64, pt: u64, ps: f64, pt_timed: u64, gt: u64, gs: f64, ttft: f64, ttft_n: u64, last: f64, est: bool }
    let mut aggs: Vec<(String, Agg)> = Vec::new();
    for c in &app.done {
        let i = match aggs.iter().position(|a| a.0 == c.model) {
            Some(i) => i,
            None => { aggs.push((c.model.clone(), Agg::default())); aggs.len() - 1 }
        };
        let a = &mut aggs[i].1;
        a.n += 1;
        a.pt += c.prompt_tokens;
        if let Some(tps) = c.prompt_tps.filter(|t| *t > 0.0) {
            a.ps += c.prompt_tokens as f64 / tps;
            a.pt_timed += c.prompt_tokens;
        }
        a.gt += c.gen_tokens;
        if c.gen_tps > 0.0 { a.gs += c.gen_tokens as f64 / c.gen_tps; }
        if let Some(t) = c.ttft { a.ttft += t; a.ttft_n += 1; }
        a.last = c.gen_tps;
        a.est |= c.source == Source::Stats;
    }
    let header = Row::new(["Model", "Reqs", "In tok/s", "Out tok/s", "Last out", "TTFT", "In tok", "Out tok"])
        .style(Style::default().fg(Color::Yellow).bold());
    let rows: Vec<Row> = aggs.iter().map(|(m, a)| {
        let color = app.colors.get(m).copied().unwrap_or(Color::White);
        let rate = |tok: u64, secs: f64| if secs > 0.0 { format!("{:.1}", tok as f64 / secs) } else { "-".into() };
        Row::new(vec![
            Cell::from(Span::styled(app.name(m), Style::default().fg(color))),
            Cell::from(a.n.to_string()),
            Cell::from(format!("{}{}", if a.est { "~" } else { "" }, rate(a.pt_timed, a.ps))),
            Cell::from(rate(a.gt, a.gs)),
            Cell::from(format!("{:.1}", a.last)),
            Cell::from(if a.ttft_n > 0 { format!("{:.2}s", a.ttft / a.ttft_n as f64) } else { "-".into() }),
            Cell::from(fmt_tokens(a.pt)),
            Cell::from(fmt_tokens(a.gt)),
        ])
    }).collect();
    let table = Table::new(rows, [
        Constraint::Fill(1), Constraint::Length(5), Constraint::Length(9), Constraint::Length(10),
        Constraint::Length(9), Constraint::Length(7), Constraint::Length(7), Constraint::Length(7),
    ])
    .header(header)
    .block(panel("Per model (session)", Color::Yellow));
    f.render_widget(table, area);
}

fn draw_recent(f: &mut Frame, area: Rect, app: &App) {
    let now = unix_now();
    let header = Row::new(["When", "Model", "In", "In t/s", "Out", "Out t/s", "TTFT"])
        .style(Style::default().fg(Color::Yellow).bold());
    let visible = area.height.saturating_sub(3) as usize;
    let rows: Vec<Row> = app.done.iter().rev().take(visible).enumerate().map(|(i, c)| {
        // Older rows fade towards the background, like btop's process list.
        let k = 0.75 * i as f64 / visible.max(2).saturating_sub(1) as f64;
        let color = fade(app.colors.get(&c.model).copied().unwrap_or(TEXT), k);
        let est = if c.source == Source::Stats { "~" } else { "" };
        Row::new(vec![
            Cell::from(fmt_ago(now - unix(c.at))),
            Cell::from(Span::styled(app.name(&c.model), Style::default().fg(color))),
            Cell::from(fmt_tokens(c.prompt_tokens)),
            Cell::from(c.prompt_tps.map(|v| format!("{est}{v:.0}")).unwrap_or_else(|| "-".into())),
            Cell::from(fmt_tokens(c.gen_tokens)),
            Cell::from(format!("{:.1}", c.gen_tps)),
            Cell::from(c.ttft.map(|t| format!("{t:.2}s")).unwrap_or_else(|| "-".into())),
        ])
        .style(Style::default().fg(fade(TEXT, k)))
    }).collect();
    let table = Table::new(rows, [
        Constraint::Length(8), Constraint::Fill(1), Constraint::Length(6), Constraint::Length(7),
        Constraint::Length(6), Constraint::Length(7), Constraint::Length(7),
    ])
    .header(header)
    .block(panel("Recent requests", Color::LightMagenta));
    f.render_widget(table, area);
}
