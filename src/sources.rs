//! Background collectors. Each runs on its own thread and sends `Msg`s to the UI.
//!
//! - llama.cpp timing lines from `lms log stream -s runtime` (exact prompt and
//!   generation speed for GGUF models)
//! - prediction stats from `lms log stream -s model --stats` (all engines,
//!   used for non-GGUF models)
//! - `lms ps` for loaded models, `ioreg`/`vm_stat` for GPU and memory

use regex::Regex;
use serde_json::Value;
use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, SystemTime};

#[derive(Clone, Copy, PartialEq)]
pub enum Source {
    LlamaCpp,
    Stats,
}

pub struct Completed {
    pub at: SystemTime,
    pub model: String,
    /// llama.cpp task id, used to drop that request's live progress once it completes.
    pub task: Option<String>,
    pub prompt_tokens: u64,
    pub gen_tokens: u64,
    /// Prompt processing speed; for `Source::Stats` it is estimated from time to first token.
    pub prompt_tps: Option<f64>,
    pub gen_tps: f64,
    pub ttft: Option<f64>,
    pub source: Source,
}

pub struct LoadedModel {
    pub id: String,
    pub name: String,
    pub format: String,
    pub quant: String,
    pub size: u64,
    pub ctx: u64,
    pub status: String,
    pub parallel: u64,
    pub queued: u64,
    /// Unix time in ms.
    pub last_used: u64,
}

pub struct SysSample {
    pub gpu: f64,
    pub gpu_mem: u64,
    pub mem_used: u64,
    pub mem_total: u64,
}

pub enum Msg {
    Completed(Completed),
    /// Prompt processing progress; `elapsed` is seconds since the prompt started.
    Prefill { model: String, task: String, progress: f64, tps: f64, elapsed: f64 },
    /// Generation progress, logged by llama.cpp every 3 s: tokens so far, average and last-3s tok/s.
    Decode { model: String, task: String, n_gen: u64, tps: f64, tps_3s: f64 },
    Loaded(Vec<LoadedModel>),
    Sys(SysSample),
}

/// Child processes to kill on exit.
pub type Children = Arc<Mutex<Vec<Child>>>;

fn lms() -> String {
    let p = format!("{}/.lmstudio/bin/lms", std::env::var("HOME").unwrap_or_default());
    if std::path::Path::new(&p).exists() { p } else { "lms".into() }
}

pub fn start(tx: Sender<Msg>, children: Children) {
    let (t, c) = (tx.clone(), children.clone());
    thread::spawn(move || stream(&["log", "stream", "-s", "runtime", "--json"], &t, &c, RuntimeParser::new()));
    let (t, c) = (tx.clone(), children);
    thread::spawn(move || stream(&["log", "stream", "-s", "model", "--stats", "--json"], &t, &c, StatsParser));
    let t = tx.clone();
    thread::spawn(move || loop {
        if let Some(models) = loaded_models() {
            if t.send(Msg::Loaded(models)).is_err() { return; }
        }
        thread::sleep(Duration::from_secs(2));
    });
    thread::spawn(move || {
        let mem_total = sysctl_u64("hw.memsize");
        loop {
            if tx.send(Msg::Sys(sys_sample(mem_total))).is_err() { return; }
            thread::sleep(Duration::from_secs(1));
        }
    });
}

trait Parser {
    fn line(&mut self, v: &Value, tx: &Sender<Msg>);
}

/// Runs an `lms log stream` command forever, restarting it if it exits
/// (e.g. LM Studio was quit and reopened).
fn stream(args: &[&str], tx: &Sender<Msg>, children: &Children, mut parser: impl Parser) {
    loop {
        let spawned = Command::new(lms()).args(args).stdout(Stdio::piped()).stderr(Stdio::null()).spawn();
        if let Ok(mut child) = spawned {
            let stdout = child.stdout.take().unwrap();
            {
                let mut kids = children.lock().unwrap();
                kids.retain_mut(|k| matches!(k.try_wait(), Ok(None)));
                kids.push(child);
            }
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                if let Ok(v) = serde_json::from_str::<Value>(&line) {
                    parser.line(&v, tx);
                }
            }
        }
        thread::sleep(Duration::from_secs(2));
    }
}

const TPS: &str = r"([\d.]+) ms /\s*(\d+) tokens \(\s*[\d.]+ ms per token,\s*([\d.]+) tokens per second\)";

struct RuntimeParser {
    prompt: Regex,
    eval: Regex,
    prefill: Regex,
    decode: Regex,
    /// (model, task) -> (prompt ms, prompt tokens, prompt tok/s), waiting for the matching eval line.
    pending: HashMap<(String, String), (f64, u64, f64)>,
}

impl RuntimeParser {
    fn new() -> Self {
        Self {
            prompt: Regex::new(&format!(r"task (\d+) \|\s+prompt eval time =\s*{TPS}")).unwrap(),
            eval: Regex::new(&format!(r"task (\d+) \|\s+eval time =\s*{TPS}")).unwrap(),
            prefill: Regex::new(r"task (\d+) \| prompt processing, n_tokens =\s*\d+, progress =\s*([\d.]+), t =\s*([\d.]+) s /\s*([\d.]+) tokens per second").unwrap(),
            decode: Regex::new(r"task (\d+) \| n_gen =\s*(\d+), tg =\s*([\d.]+) t/s, tg_3s =\s*([\d.]+) t/s").unwrap(),
            pending: HashMap::new(),
        }
    }
}

impl Parser for RuntimeParser {
    fn line(&mut self, v: &Value, tx: &Sender<Msg>) {
        let data = &v["data"];
        let (Some(msg), Some(model)) = (data["message"].as_str(), data["modelIdentifier"].as_str()) else { return };
        for l in msg.lines() {
            if let Some(c) = self.prefill.captures(l) {
                let (Ok(progress), Ok(elapsed), Ok(tps)) = (c[2].parse(), c[3].parse(), c[4].parse()) else { continue };
                let _ = tx.send(Msg::Prefill { model: model.into(), task: c[1].into(), progress, tps, elapsed });
            } else if let Some(c) = self.decode.captures(l) {
                let (Ok(n_gen), Ok(tps), Ok(tps_3s)) = (c[2].parse(), c[3].parse(), c[4].parse()) else { continue };
                let _ = tx.send(Msg::Decode { model: model.into(), task: c[1].into(), n_gen, tps, tps_3s });
            } else if let Some(c) = self.prompt.captures(l) {
                let (Ok(ms), Ok(n), Ok(tps)) = (c[2].parse(), c[3].parse(), c[4].parse()) else { continue };
                self.pending.insert((model.into(), c[1].into()), (ms, n, tps));
            } else if let Some(c) = self.eval.captures(l) {
                let (Ok(n), Ok(tps)) = (c[3].parse::<u64>(), c[4].parse::<f64>()) else { continue };
                let p = self.pending.remove(&(model.to_string(), c[1].to_string()));
                let _ = tx.send(Msg::Completed(Completed {
                    at: SystemTime::now(),
                    model: model.into(),
                    task: Some(c[1].to_string()),
                    prompt_tokens: p.map_or(0, |p| p.1),
                    gen_tokens: n,
                    prompt_tps: p.map(|p| p.2),
                    gen_tps: tps,
                    ttft: p.map(|p| p.0 / 1000.0),
                    source: Source::LlamaCpp,
                }));
            }
        }
    }
}

struct StatsParser;

/// Finds the first object (depth-first) that has `key`.
fn find_with<'a>(v: &'a Value, key: &str) -> Option<&'a serde_json::Map<String, Value>> {
    match v {
        Value::Object(m) if m.contains_key(key) => Some(m),
        Value::Object(m) => m.values().find_map(|x| find_with(x, key)),
        Value::Array(a) => a.iter().find_map(|x| find_with(x, key)),
        _ => None,
    }
}

impl Parser for StatsParser {
    fn line(&mut self, v: &Value, tx: &Sender<Msg>) {
        let Some(stats) = find_with(v, "tokensPerSecond") else { return };
        let Some(model) = find_with(v, "modelIdentifier").and_then(|m| m["modelIdentifier"].as_str()) else { return };
        let f = |k: &str| stats.get(k).and_then(Value::as_f64);
        let Some(gen_tps) = f("tokensPerSecond") else { return };
        let prompt_tokens = f("promptTokensCount").unwrap_or(0.0) as u64;
        let ttft = f("timeToFirstTokenSec");
        let _ = tx.send(Msg::Completed(Completed {
            at: SystemTime::now(),
            model: model.into(),
            task: None,
            prompt_tokens,
            gen_tokens: f("predictedTokensCount").unwrap_or(0.0) as u64,
            prompt_tps: ttft.filter(|t| *t > 0.0).map(|t| prompt_tokens as f64 / t),
            gen_tps,
            ttft,
            source: Source::Stats,
        }));
    }
}

fn output(cmd: &str, args: &[&str]) -> Option<String> {
    let o = Command::new(cmd).args(args).stderr(Stdio::null()).output().ok()?;
    o.status.success().then(|| String::from_utf8_lossy(&o.stdout).into_owned())
}

fn loaded_models() -> Option<Vec<LoadedModel>> {
    let v: Value = serde_json::from_str(&output(&lms(), &["ps", "--json"])?).ok()?;
    let u = |m: &Value, k: &str| m[k].as_u64().unwrap_or(0);
    let s = |m: &Value, k: &str| m[k].as_str().unwrap_or("").to_string();
    Some(
        v.as_array()?
            .iter()
            .map(|m| LoadedModel {
                id: s(m, "identifier"),
                name: s(m, "displayName"),
                format: s(m, "format"),
                quant: m["quantization"]["name"].as_str().unwrap_or("").into(),
                size: u(m, "sizeBytes"),
                ctx: u(m, "contextLength"),
                status: s(m, "status"),
                parallel: u(m, "parallel"),
                queued: u(m, "queued"),
                last_used: u(m, "lastUsedTime"),
            })
            .collect(),
    )
}

fn sysctl_u64(name: &str) -> u64 {
    output("sysctl", &["-n", name]).and_then(|s| s.trim().parse().ok()).unwrap_or(0)
}

fn sys_sample(mem_total: u64) -> SysSample {
    let io = output("ioreg", &["-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"]).unwrap_or_default();
    let io_num = |key: &str| {
        let pat = format!("\"{key}\"=");
        io.find(&pat).and_then(|i| {
            let rest = &io[i + pat.len()..];
            rest[..rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len())].parse::<u64>().ok()
        })
    };
    let vm = output("vm_stat", &[]).unwrap_or_default();
    let page = vm.split("page size of ").nth(1).and_then(|s| s.split_whitespace().next()).and_then(|s| s.parse().ok()).unwrap_or(16384u64);
    let pages = |key: &str| {
        vm.lines()
            .find(|l| l.starts_with(key))
            .and_then(|l| l.rsplit(':').next())
            .and_then(|n| n.trim().trim_end_matches('.').parse::<u64>().ok())
            .unwrap_or(0)
    };
    // Approximates Activity Monitor's "Memory Used": active + wired + compressed.
    let used = (pages("Pages active") + pages("Pages wired down") + pages("Pages occupied by compressor")) * page;
    SysSample {
        gpu: io_num("Device Utilization %").unwrap_or(0) as f64,
        gpu_mem: io_num("In use system memory").unwrap_or(0),
        mem_used: used,
        mem_total,
    }
}
