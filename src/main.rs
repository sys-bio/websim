// Don't open a console window on Windows (debug and release builds).
#![windows_subsystem = "windows"]

mod antimony;
mod model;
mod ode;
mod solvers;

use std::collections::HashMap;

use eframe::egui;
use egui::epaint::CubicBezierShape;
use egui::{Align2, Color32, FontId, Pos2, Rect, Sense, Shape, Stroke, StrokeKind, vec2};
use egui_plot::{Legend, Line, LineStyle, Plot, Points, VLine};
use model::{EXAMPLES, Model, Results};
use solvers::{Method, SolverSettings};

/// Desktop entry point: open a native window.
#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 760.0])
            .with_title("egui GUI Test"),
        centered: true,
        ..Default::default()
    };

    eframe::run_native("egui GUI Test", options, Box::new(create_app))
}

/// Web entry point: attach the app to the <canvas> in index.html.
#[cfg(target_arch = "wasm32")]
fn main() {
    use eframe::wasm_bindgen::JsCast as _;

    wasm_bindgen_futures::spawn_local(async {
        let document = eframe::web_sys::window()
            .expect("no window")
            .document()
            .expect("no document");
        let canvas = document
            .get_element_by_id("the_canvas_id")
            .expect("no element with id 'the_canvas_id'")
            .dyn_into::<eframe::web_sys::HtmlCanvasElement>()
            .expect("'the_canvas_id' is not a <canvas>");

        let result = eframe::WebRunner::new()
            .start(canvas, eframe::WebOptions::default(), Box::new(create_app))
            .await;

        // Replace the "Loading…" message with the app, or with an error.
        if let Some(loading) = document.get_element_by_id("loading_text") {
            match result {
                Ok(()) => loading.remove(),
                Err(e) => loading.set_inner_html(&format!("The app crashed: {e:?}")),
            }
        }
    });
}

/// Shared by both entry points: set up styling and create the app.
fn create_app(
    cc: &eframe::CreationContext<'_>,
) -> Result<Box<dyn eframe::App>, Box<dyn std::error::Error + Send + Sync>> {
    set_font_sizes(&cc.egui_ctx);
    Ok(Box::<MyApp>::default())
}

/// Make all the built-in text styles larger than egui's defaults.
fn set_font_sizes(ctx: &egui::Context) {
    use egui::{FontFamily::Proportional, FontFamily::Monospace, FontId, TextStyle};

    ctx.all_styles_mut(|style| {
        style.text_styles = [
            (TextStyle::Heading, FontId::new(26.0, Proportional)),
            (TextStyle::Body, FontId::new(18.0, Proportional)),
            (TextStyle::Button, FontId::new(18.0, Proportional)),
            (TextStyle::Small, FontId::new(14.0, Proportional)),
            (TextStyle::Monospace, FontId::new(16.0, Monospace)),
        ]
        .into();
    });
}

/// Event markers drawn on the time-course plot, at most.
const MAX_EVENT_LINES: usize = 500;

/// The width of `text` when drawn in `style`.
fn text_width(ui: &egui::Ui, text: &str, style: egui::TextStyle) -> f32 {
    egui::WidgetText::from(text)
        .into_galley(ui, Some(egui::TextWrapMode::Extend), f32::INFINITY, style)
        .size()
        .x
}

/// Width of the widest of `names` in the body font, so labels can form a column.
fn label_column_width<'a>(ui: &egui::Ui, names: impl Iterator<Item = &'a str>) -> f32 {
    names
        .map(|name| text_width(ui, name, egui::TextStyle::Body))
        .fold(0.0, f32::max)
}

/// Show a model value compactly, with 4 significant digits, so that the number
/// box next to each slider never needs more room than [`slider_value_room`] gives it.
fn format_value(v: f64) -> String {
    fn trim_zeros(s: &str) -> &str {
        if s.contains('.') {
            s.trim_end_matches('0').trim_end_matches('.')
        } else {
            s
        }
    }
    let size = v.abs();
    if size == 0.0 || !v.is_finite() {
        format!("{v}")
    } else if size >= 1e5 || size < 1e-3 {
        let s = format!("{v:.3e}"); // e.g. 3.000e7
        match s.split_once('e') {
            Some((mantissa, exponent)) => format!("{}e{exponent}", trim_zeros(mantissa)),
            None => s,
        }
    } else {
        let decimals = (3 - size.log10().floor() as i32).max(0) as usize;
        trim_zeros(&format!("{v:.decimals$}")).to_owned()
    }
}

/// Room to leave at the end of a slider row for its number box: the width of the
/// widest number [`format_value`] produces, plus the box's padding and spacing.
///
/// This must be a constant, not the width of the current value: the side panel
/// grows to fit its contents, so a row that overflows makes the panel wider, which
/// makes the slider wider, which overflows again — the panel would grow forever.
fn slider_value_room(ui: &egui::Ui) -> f32 {
    let widest = text_width(ui, "-8.888e-308", egui::TextStyle::Button);
    let spacing = ui.spacing();
    (widest + 2.0 * spacing.button_padding.x).max(spacing.interact_size.x) + spacing.item_spacing.x + 4.0
}

/// One row: the name in a column of width `label_width`, then `add_contents`.
/// Sliders added in `add_contents` stretch to fill the rest of the row.
fn labeled_row(ui: &mut egui::Ui, label_width: f32, name: &str, add_contents: impl FnOnce(&mut egui::Ui)) {
    ui.horizontal(|ui| {
        let size = vec2(label_width, ui.spacing().interact_size.y);
        ui.allocate_ui_with_layout(size, egui::Layout::left_to_right(egui::Align::Center), |ui| {
            ui.set_min_width(label_width);
            ui.label(name);
        });
        let room = slider_value_room(ui);
        ui.spacing_mut().slider_width = (ui.available_width() - room).max(60.0);
        add_contents(ui);
    });
}

/// A slider for a model value. Its range is based on the value written in the
/// model text, so it doesn't shift while dragging; values outside the range can be typed.
fn value_slider(ui: &mut egui::Ui, label_width: f32, value: &mut f64, default: f64, name: &str) {
    let range = if default > 0.0 {
        0.0..=3.0 * default
    } else if default < 0.0 {
        3.0 * default..=0.0
    } else {
        0.0..=1.0
    };
    labeled_row(ui, label_width, name, |ui| {
        ui.add(
            egui::Slider::new(value, range)
                .clamping(egui::SliderClamping::Never)
                .custom_formatter(|v, _| format_value(v)),
        );
    });
}

/// A logarithmic slider for a solver tolerance, shown like 1e-6.
fn tolerance_slider(ui: &mut egui::Ui, label_width: f32, name: &str, value: &mut f64) {
    labeled_row(ui, label_width, name, |ui| {
        ui.add(
            egui::Slider::new(value, 1e-14..=1e-2)
                .logarithmic(true)
                .custom_formatter(|v, _| format!("{v:.0e}"))
                .custom_parser(|s| s.trim().parse().ok()),
        );
    });
}

/// The solver controls in the side panel.
fn solver_controls(ui: &mut egui::Ui, label_width: f32, settings: &mut SolverSettings) {
    labeled_row(ui, label_width, "solver", |ui| {
        egui::ComboBox::from_id_salt("solver")
            .selected_text(settings.method.label())
            .show_ui(ui, |ui| {
                for method in Method::ALL {
                    ui.selectable_value(&mut settings.method, method, method.label());
                }
            });
    });
    if settings.method.is_adaptive() {
        tolerance_slider(ui, label_width, "rel. tol", &mut settings.rtol);
        tolerance_slider(ui, label_width, "abs. tol", &mut settings.atol);
    } else {
        labeled_row(ui, label_width, "steps", |ui| {
            ui.add(egui::Slider::new(&mut settings.rk4_steps, 100..=200_000).logarithmic(true));
        });
    }
}

/// A drop-down list for choosing one of the plottable quantities.
fn output_combo(ui: &mut egui::Ui, id: &str, selected: &mut usize, names: &[String]) {
    egui::ComboBox::from_id_salt(id)
        .selected_text(names[*selected].as_str())
        .show_ui(ui, |ui| {
            for (i, name) in names.iter().enumerate() {
                ui.selectable_value(selected, i, name.as_str());
            }
        });
}

/// Whether quantity `i` is ticked for plotting. Species are plotted unless
/// the user unticked them; rates and rules only once the user ticks them.
fn is_plotted(choice: &HashMap<String, bool>, res: &Results, i: usize) -> bool {
    let is_species = i < res.species_count;
    choice.get(&res.names[i]).copied().unwrap_or(is_species)
}

/// A fixed colour per quantity, so a line keeps its colour when others are toggled.
fn series_color(i: usize) -> Color32 {
    const PALETTE: [(u8, u8, u8); 10] = [
        (31, 119, 180),
        (255, 127, 14),
        (44, 160, 44),
        (214, 39, 40),
        (148, 103, 189),
        (140, 86, 75),
        (227, 119, 194),
        (127, 127, 127),
        (188, 189, 34),
        (23, 190, 207),
    ];
    let (r, g, b) = PALETTE[i % PALETTE.len()];
    Color32::from_rgb(r, g, b)
}

/// One freehand line drawn by the user with the mouse.
/// Points are stored relative to the canvas's top-left corner,
/// so the drawing stays put if the window is moved or resized.
struct PenStroke {
    points: Vec<Pos2>,
    stroke: Stroke,
}

/// Which page is shown in the main area.
#[derive(Clone, Copy, PartialEq)]
enum Tab {
    Simulation,
    Canvas,
}

/// All application state lives in this struct. egui is "immediate mode":
/// `ui` is called every frame and rebuilds the whole UI from this state.
struct MyApp {
    tab: Tab,
    /// The model as typed by the user.
    model_text: String,
    /// The last version of the text that parsed without errors.
    model: Option<Model>,
    parse_error: Option<String>,
    t_end: f64,
    results: Results,
    /// What `results` were computed from, so we only re-simulate on change.
    solved_for: Option<(Model, f64, SolverSettings)>,
    solver: SolverSettings,
    /// Plot checkboxes the user has changed, by name. Species are plotted
    /// by default; reaction rates and other rules are not.
    plot_choice: HashMap<String, bool>,
    /// While set, the time-course plot keeps this y range.
    fixed_y: Option<(f64, f64)>,
    /// The y range shown last frame, captured when "Fix y-axis" is ticked.
    last_y_range: (f64, f64),
    /// Ask the time-course plot to fit its data again (after unfixing the y-axis).
    refit_time_course: bool,
    /// Which quantities are plotted against each other in the phase plane.
    phase_x: usize,
    phase_y: usize,
    name: String,
    age: u32,
    clicks: u32,
    dark_mode: bool,
    color: Color32,
    pen_width: f32,
    pen_strokes: Vec<PenStroke>,
}

impl Default for MyApp {
    fn default() -> Self {
        Self {
            tab: Tab::Simulation,
            model_text: EXAMPLES[0].1.to_owned(),
            model: Model::parse(EXAMPLES[0].1).ok(),
            parse_error: None,
            t_end: 50.0,
            results: Results::default(),
            solved_for: None,
            solver: SolverSettings::default(),
            plot_choice: HashMap::new(),
            fixed_y: None,
            last_y_range: (0.0, 1.0),
            refit_time_course: false,
            phase_x: 0,
            phase_y: 1,
            name: "World".to_owned(),
            age: 42,
            clicks: 0,
            dark_mode: true,
            color: Color32::from_rgb(100, 150, 250),
            pen_width: 3.0,
            pen_strokes: Vec::new(),
        }
    }
}

impl eframe::App for MyApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        ctx.set_visuals(if self.dark_mode {
            egui::Visuals::dark()
        } else {
            egui::Visuals::light()
        });

        egui::Panel::top("menu_bar").show(ui, |ui| {
            egui::MenuBar::new().ui(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("Reset").clicked() {
                        *self = MyApp::default();
                        ui.close();
                    }
                    // A web page can't close its own browser tab, so desktop only.
                    #[cfg(not(target_arch = "wasm32"))]
                    if ui.button("Quit").clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });
                ui.checkbox(&mut self.dark_mode, "Dark mode");
                ui.separator();
                ui.selectable_value(&mut self.tab, Tab::Simulation, "ODE simulation");
                ui.selectable_value(&mut self.tab, Tab::Canvas, "Canvas");
            });
        });

        egui::Panel::left("controls")
            .resizable(true)
            .default_size(440.0)
            .show(ui, |ui| match self.tab {
                Tab::Simulation => self.model_controls(ui),
                Tab::Canvas => self.controls(ui),
            });

        egui::CentralPanel::default().show(ui, |ui| match self.tab {
            Tab::Simulation => self.simulation(ui),
            Tab::Canvas => self.canvas(ui),
        });
    }
}

impl MyApp {
    /// Re-read the model text. If it has an error, the last good model stays on screen.
    fn reparse(&mut self) {
        match Model::parse(&self.model_text) {
            Ok(mut model) => {
                if let Some(old) = &self.model {
                    model.keep_values_from(old);
                }
                self.model = Some(model);
                self.parse_error = None;
            }
            Err(e) => self.parse_error = Some(e.to_string()),
        }
    }

    fn load_example(&mut self, text: &str) {
        self.model_text = text.to_owned();
        self.model = None; // don't carry slider values over from another model
        self.reparse();
        self.phase_x = 0;
        self.phase_y = 1;
        // A different model needs a different scale.
        if self.fixed_y.take().is_some() {
            self.refit_time_course = true;
        }
    }

    /// Side panel for the ODE tab: the model editor, then a slider per value.
    fn model_controls(&mut self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Model (Antimony)");
                ui.menu_button("Examples", |ui| {
                    for (name, text) in EXAMPLES {
                        if ui.button(*name).clicked() {
                            self.load_example(text);
                            ui.close();
                        }
                    }
                });
            });

            let editor = egui::TextEdit::multiline(&mut self.model_text)
                .code_editor()
                .desired_rows(16)
                .desired_width(f32::INFINITY);
            if ui.add(editor).changed() {
                self.reparse();
            }
            if let Some(error) = &self.parse_error {
                ui.colored_label(ui.visuals().error_fg_color, error);
            }
            ui.small(
                "J1: $Xo -> 2 S1; k*Xo   reaction ($ = boundary species)\n\
                 v := k*S1   rule      x' = -k*x   rate rule\n\
                 k = 1.5   parameter or initial value      // comment\n\
                 E1: at (time > 10 && S1 < 2): k = 2, S1 = 0   event",
            );
            ui.separator();

            let Some(model) = &mut self.model else { return };

            let end_time = "end time";
            let other_labels = [end_time, "solver", "rel. tol", "abs. tol", "steps"];
            let label_width =
                label_column_width(ui, model.symbols().map(|s| s.name.as_str()).chain(other_labels));

            ui.strong("Parameters");
            for p in model.params_mut() {
                value_slider(ui, label_width, &mut p.value, p.default, &p.name);
            }
            ui.strong("Initial values");
            for s in &mut model.species {
                value_slider(ui, label_width, &mut s.value, s.default, &s.name);
            }
            ui.strong("Simulation");
            labeled_row(ui, label_width, end_time, |ui| {
                ui.add(
                    egui::Slider::new(&mut self.t_end, 1.0..=200.0)
                        .clamping(egui::SliderClamping::Never)
                        .custom_formatter(|v, _| format_value(v)),
                );
            });
            solver_controls(ui, label_width, &mut self.solver);

            if ui.button("Reset values").clicked() {
                model.reset_values();
            }
            ui.add_space(8.0);
            ui.small("Plots: drag to pan, scroll to zoom, double-click to reset the view.");
        });
    }

    /// The ODE tab: a time course of the chosen species and rates, then a phase plane.
    fn simulation(&mut self, ui: &mut egui::Ui) {
        let Some(model) = &self.model else {
            ui.label("Type a model on the left to see it simulated.");
            return;
        };

        // Only re-run the solver when the model, a value, the end time or the solver has changed.
        let up_to_date = matches!(&self.solved_for,
            Some((m, t, s)) if m == model && *t == self.t_end && *s == self.solver);
        if !up_to_date {
            self.results = model.run(self.t_end, &self.solver);
            self.solved_for = Some((model.clone(), self.t_end, self.solver));
        }
        let res = &self.results;

        if let Some(reason) = &res.stopped_early {
            ui.colored_label(ui.visuals().warn_fg_color, reason);
        }

        // Checkboxes choosing what to plot: species first, then rates and rules.
        ui.horizontal_wrapped(|ui| {
            ui.label("Plot:");
            for (i, name) in res.names.iter().enumerate() {
                if i == res.species_count {
                    ui.separator();
                    ui.label("Rates:");
                }
                let mut shown = is_plotted(&self.plot_choice, res, i);
                if ui.checkbox(&mut shown, name.as_str()).changed() {
                    self.plot_choice.insert(name.clone(), shown);
                }
            }
        });

        ui.horizontal(|ui| {
            let mut fixed = self.fixed_y.is_some();
            if ui
                .checkbox(&mut fixed, "Fix y-axis")
                .on_hover_text("Keep the current scale while you move the sliders")
                .changed()
            {
                self.fixed_y = fixed.then_some(self.last_y_range);
                self.refit_time_course = !fixed;
            }
            if let Some((low, high)) = &mut self.fixed_y {
                let speed = (*high - *low).abs().max(1e-9) * 0.005;
                ui.label("from");
                ui.add(egui::DragValue::new(low).speed(speed));
                ui.label("to");
                ui.add(egui::DragValue::new(high).speed(speed));
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let events = match res.events.len() {
                    0 => String::new(),
                    1 => ", 1 event".to_owned(),
                    n => format!(", {n} events"),
                };
                ui.small(format!(
                    "{} steps, {} rate evaluations{events}",
                    res.steps, res.rate_evaluations
                ));
            });
        });

        let n = res.names.len();
        let show_phase_plane = n >= 2;
        let height = if show_phase_plane {
            ui.available_height() * 0.55
        } else {
            ui.available_height()
        };

        let fixed_y = self.fixed_y;
        let refit = std::mem::take(&mut self.refit_time_course);
        let plotted: Vec<usize> = (0..n).filter(|&i| is_plotted(&self.plot_choice, res, i)).collect();

        let time_course = Plot::new("time_course")
            .height(height)
            .legend(Legend::default())
            .x_axis_label("time")
            .show(ui, |plot_ui| {
                if let Some((low, high)) = fixed_y {
                    plot_ui.set_plot_bounds_y(low.min(high)..=low.max(high));
                }
                if refit {
                    plot_ui.set_auto_bounds(true);
                }
                // A dotted vertical line where each event fired (at most a few hundred).
                for (t, name) in res.events.iter().take(MAX_EVENT_LINES) {
                    plot_ui.vline(
                        VLine::new(name.as_str(), *t)
                            .color(Color32::GRAY)
                            .style(LineStyle::dashed_dense())
                            .width(1.0),
                    );
                }
                for &i in &plotted {
                    let mut line = Line::new(res.names[i].as_str(), res.series(i))
                        .color(series_color(i))
                        .width(2.0);
                    if i >= res.species_count {
                        line = line.style(LineStyle::dashed_loose()); // rates are dashed
                    }
                    plot_ui.line(line);
                }
            });
        let bounds = time_course.transform.bounds();
        self.last_y_range = (bounds.min()[1], bounds.max()[1]);

        if !show_phase_plane {
            return;
        }

        ui.add_space(4.0);
        self.phase_x = self.phase_x.min(n - 1);
        self.phase_y = self.phase_y.min(n - 1);
        ui.horizontal(|ui| {
            ui.label("Phase plane:");
            output_combo(ui, "phase_y", &mut self.phase_y, &res.names);
            ui.label("against");
            output_combo(ui, "phase_x", &mut self.phase_x, &res.names);
        });

        let (ix, iy) = (self.phase_x, self.phase_y);
        let trajectory: Vec<[f64; 2]> = res.columns[ix]
            .iter()
            .zip(&res.columns[iy])
            .map(|(x, y)| [*x, *y])
            .collect();
        let start = trajectory.first().copied();

        Plot::new("phase_plane")
            .legend(Legend::default())
            .x_axis_label(res.names[ix].as_str())
            .y_axis_label(res.names[iy].as_str())
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new("Trajectory", trajectory)
                        .color(Color32::from_rgb(110, 140, 230))
                        .width(2.0),
                );
                if let Some(start) = start {
                    plot_ui.points(Points::new("Start", vec![start]).radius(5.0));
                }
            });
    }

    /// The widgets in the left-hand side panel.
    fn controls(&mut self, ui: &mut egui::Ui) {
        ui.heading("Controls");
        ui.separator();

        ui.horizontal(|ui| {
            ui.label("Your name:");
            ui.text_edit_singleline(&mut self.name);
        });

        ui.add(egui::Slider::new(&mut self.age, 0..=120).text("age"));

        if ui.button("Increment age").clicked() {
            self.age += 1;
        }

        ui.label(format!("Hello '{}', age {}", self.name, self.age));
        ui.separator();

        ui.horizontal(|ui| {
            if ui.button("Click me!").clicked() {
                self.clicks += 1;
            }
            ui.label(format!("Clicked {} times", self.clicks));
        });
        ui.separator();

        ui.label("Pen (drag on the canvas to draw):");
        ui.horizontal(|ui| {
            ui.label("Colour:");
            ui.color_edit_button_srgba(&mut self.color);
        });
        ui.add(egui::Slider::new(&mut self.pen_width, 1.0..=20.0).text("width"));
        if ui.button("Clear drawing").clicked() {
            self.pen_strokes.clear();
        }
    }

    /// A drawing area: some fixed shapes plus whatever the user draws.
    fn canvas(&mut self, ui: &mut egui::Ui) {
        // Reserve all remaining space and get a Painter that is clipped to it.
        // Sense::drag() means we get mouse-drag events for this area.
        let (response, painter) = ui.allocate_painter(ui.available_size(), Sense::drag());
        let rect = response.rect;
        let text_color = ui.visuals().text_color();

        // Background and border.
        painter.rect_filled(rect, 0.0, ui.visuals().extreme_bg_color);
        painter.rect_stroke(
            rect,
            0.0,
            ui.visuals().widgets.noninteractive.bg_stroke,
            StrokeKind::Inside,
        );

        // Place things by fraction of the canvas size (0.0..1.0),
        // so the drawing scales when the window is resized.
        let at = |fx: f32, fy: f32| rect.min + vec2(fx * rect.width(), fy * rect.height());
        let outline = Stroke::new(2.0, text_color);

        // Lines: a solid one and a dashed one.
        painter.line_segment([at(0.05, 0.10), at(0.35, 0.35)], Stroke::new(4.0, Color32::RED));
        painter.extend(Shape::dashed_line(
            &[at(0.05, 0.35), at(0.35, 0.10)],
            Stroke::new(2.0, Color32::GRAY),
            12.0,
            6.0,
        ));

        // A rounded rectangle, filled and outlined.
        let r = Rect::from_min_max(at(0.42, 0.10), at(0.68, 0.35));
        painter.rect_filled(r, 10.0, Color32::from_rgb(80, 160, 90));
        painter.rect_stroke(r, 10.0, outline, StrokeKind::Outside);

        // A circle that uses the colour chosen in the side panel.
        let radius = 0.12 * rect.width().min(rect.height());
        painter.circle_filled(at(0.84, 0.23), radius, self.color);
        painter.circle_stroke(at(0.84, 0.23), radius, outline);

        // A filled triangle.
        painter.add(Shape::convex_polygon(
            vec![at(0.20, 0.50), at(0.33, 0.85), at(0.07, 0.85)],
            Color32::from_rgb(230, 160, 40),
            outline,
        ));

        // A smooth Bezier curve through four control points.
        painter.add(CubicBezierShape::from_points_stroke(
            [at(0.42, 0.85), at(0.55, 0.35), at(0.75, 1.05), at(0.95, 0.50)],
            false,
            Color32::TRANSPARENT,
            Stroke::new(4.0, Color32::from_rgb(160, 90, 220)),
        ));

        painter.text(
            at(0.5, 0.95),
            Align2::CENTER_CENTER,
            "Drawn with egui's Painter",
            FontId::proportional(16.0),
            text_color,
        );

        // Freehand drawing: start a new stroke when a drag begins,
        // then add the mouse position to it every frame while dragging.
        if response.drag_started() {
            self.pen_strokes.push(PenStroke {
                points: Vec::new(),
                stroke: Stroke::new(self.pen_width, self.color),
            });
        }
        if response.dragged() {
            if let (Some(pos), Some(current)) =
                (response.interact_pointer_pos(), self.pen_strokes.last_mut())
            {
                let p = (pos - rect.min).to_pos2();
                if current.points.last() != Some(&p) {
                    current.points.push(p);
                }
            }
        }

        for pen in &self.pen_strokes {
            if pen.points.len() >= 2 {
                let points = pen.points.iter().map(|p| rect.min + p.to_vec2()).collect();
                painter.add(Shape::line(points, pen.stroke));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::format_value;

    #[test]
    fn values_are_formatted_compactly() {
        assert_eq!(format_value(3e7), "3e7");
        assert_eq!(format_value(1e4), "10000");
        assert_eq!(format_value(0.04), "0.04");
        assert_eq!(format_value(26.666666), "26.67");
        assert_eq!(format_value(8.0 / 3.0), "2.667");
        assert_eq!(format_value(1.5e-6), "1.5e-6");
        assert_eq!(format_value(0.0), "0");
        assert_eq!(format_value(-12.5), "-12.5");
    }

    /// The slider rows reserve room for "-8.888e-308"; nothing may be longer.
    #[test]
    fn formatted_values_fit_the_reserved_room() {
        let longest = "-8.888e-308".len();
        for exponent in -300..=300 {
            for mantissa in [1.0, 1.2345, 9.9999, -1.0, -9.8765] {
                let v = mantissa * 10f64.powi(exponent);
                let s = format_value(v);
                assert!(s.len() <= longest, "{v} formats as {s}");
            }
        }
    }
}
