//! btop-style braille graph: each dot column is one sample, filled from the baseline,
//! coloured by a vertical gradient.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Color;
use ratatui::widgets::Widget;

/// Low → mid → high colour stops.
pub type Gradient = [(u8, u8, u8); 3];

/// Green → yellow → red, bright enough for thin braille dots on a dark background.
pub const HEAT: Gradient = [(0x5f, 0xff, 0x8f), (0xff, 0xe6, 0x4d), (0xff, 0x55, 0x55)];

pub fn gradient(g: &Gradient, t: f64) -> Color {
    let t = t.clamp(0.0, 1.0);
    let (a, b, k) = if t < 0.5 { (g[0], g[1], t * 2.0) } else { (g[1], g[2], (t - 0.5) * 2.0) };
    let mix = |x: u8, y: u8| (x as f64 + (y as f64 - x as f64) * k).round() as u8;
    Color::Rgb(mix(a.0, b.0), mix(a.1, b.1), mix(a.2, b.2))
}

pub enum Fill<'a> {
    /// Coloured by height.
    Gradient(&'a Gradient),
    Solid(Color),
}

pub struct Graph<'a> {
    /// One value per dot column (2 per cell), 0..=1. Must be `area.width * 2` long.
    pub values: &'a [f64],
    pub fill: Fill<'a>,
}

// Braille dot bits, ordered from the bottom up: [left column, right column].
const BITS: [[u8; 4]; 2] = [[0x40, 0x04, 0x02, 0x01], [0x80, 0x20, 0x10, 0x08]];

impl Widget for Graph<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let rows = area.height as usize;
        // Height of each sample in dots; any non-zero value shows at least one dot.
        let dots: Vec<usize> = self.values.iter().map(|v| {
            let d = (v.clamp(0.0, 1.0) * (rows * 4) as f64).round() as usize;
            if *v > 0.0 { d.max(1) } else { 0 }
        }).collect();
        for cx in 0..area.width as usize {
            for r in 0..rows {
                // r counts cell rows from the bottom.
                let mut ch = 0u8;
                for side in 0..2 {
                    let h = dots.get(cx * 2 + side).copied().unwrap_or(0);
                    for (i, bit) in BITS[side].iter().enumerate() {
                        if h > r * 4 + i {
                            ch |= bit;
                        }
                    }
                }
                if ch == 0 {
                    continue;
                }
                let color = match self.fill {
                    Fill::Gradient(g) => gradient(g, (r as f64 + 0.5) / rows as f64),
                    Fill::Solid(c) => c,
                };
                let y = area.y + area.height - 1 - r as u16;
                if let Some(cell) = buf.cell_mut((area.x + cx as u16, y)) {
                    cell.set_char(char::from_u32(0x2800 + ch as u32).unwrap()).set_fg(color);
                }
            }
        }
    }
}
