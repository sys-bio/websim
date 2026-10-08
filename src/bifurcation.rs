//! The bifurcation view: equilibrium branches continued in one parameter with
//! the `bifurcata` library, drawn as they grow.
//!
//! The engine is an iterator, so a few continuation steps are taken each frame
//! and the plot redraws in between — the browser has no threads, and the user
//! can stop a run at any point. Stable stretches are solid, unstable ones
//! dashed; folds (LP), branch points (BP) and Hopf points (H) are marked.

use bifurcata::antimony::AntimonyProblem;
use bifurcata::continuation::{ContinuationEngine, ContinuationOptions};
use bifurcata::equilibrium::EquilibriumCurve;
use bifurcata::problem::BifurcationProblem;
use bifurcata::runspec::RunSpec;
use bifurcata::types::{BifurcationKind, Branch, StepResult};
use eframe::egui;
use egui::Color32;
use egui_plot::{Legend, Line, LineStyle, Plot, PlotPoint, Points, Text};
use websim_model::model::Model;

type Engine = ContinuationEngine<EquilibriumCurve<AntimonyProblem>>;

/// Continuation steps per frame: enough to grow quickly, few enough to stay responsive.
const STEPS_PER_FRAME: usize = 8;

/// A run in progress or finished.
struct Run {
    /// The model (and slider values) it was computed for.
    model: Model,
    parameter_name: String,
    /// The index of the active parameter in the problem's parameter list.
    active: usize,
    /// How many of the problem's parameters belong to the model; the rest are
    /// conserved totals.
    model_parameters: usize,
    /// Branches already finished.
    done: Vec<Branch>,
    /// The branch growing now, then those still to start.
    current: Option<Engine>,
    pending: Vec<Engine>,
    stopped: bool,
}

impl Run {
    fn running(&self) -> bool {
        self.current.is_some() && !self.stopped
    }

    fn branches(&self) -> impl Iterator<Item = &Branch> {
        self.done.iter().chain(self.current.iter().map(|e| e.branch()))
    }

    /// Advance the current branch by up to `steps` points, moving on to the
    /// next direction when it ends.
    fn advance(&mut self, steps: usize) {
        for _ in 0..steps {
            let Some(engine) = &mut self.current else { return };
            if engine.step() != StepResult::Ok {
                let engine = self.current.take().unwrap();
                self.done.push(engine.into_branch());
                self.current = self.pending.pop();
            }
        }
    }
}

pub struct BifurcationView {
    /// The chosen parameter, by name, so it survives edits to the model.
    parameter: Option<String>,
    range: (f64, f64),
    /// The plotted species, by name.
    plotted: Option<String>,
    max_step: f64,
    /// From the model's `[bifurcation]` block, when it gives them.
    initial_step: Option<f64>,
    max_points: Option<usize>,
    run: Option<Run>,
    error: Option<String>,
}

impl Default for BifurcationView {
    fn default() -> Self {
        Self { parameter: None, range: (0.0, 1.0), plotted: None, max_step: 0.1, initial_step: None, max_points: None, run: None, error: None }
    }
}

/// A sensible default range for a parameter currently at `value`.
fn default_range(value: f64) -> (f64, f64) {
    if value > 0.0 {
        (0.0, 3.0 * value)
    } else if value < 0.0 {
        (3.0 * value, 0.0)
    } else {
        (0.0, 1.0)
    }
}

fn kind_colour(kind: BifurcationKind) -> Color32 {
    match kind {
        BifurcationKind::Fold => Color32::from_rgb(230, 160, 30),
        BifurcationKind::Hopf => Color32::from_rgb(220, 70, 70),
        BifurcationKind::BranchPoint => Color32::from_rgb(170, 90, 220),
        _ => Color32::GRAY,
    }
}

impl BifurcationView {
    /// Forget the run (a different model was loaded).
    pub fn clear(&mut self) {
        *self = Self::default();
    }

    /// Take the parameter, range, step sizes and plotted variable from a
    /// model's `[bifurcation]` block, where it gives them.
    pub fn apply_spec(&mut self, model_text: &str) {
        let spec = RunSpec::parse(model_text);
        if let Some(p) = spec.parameter {
            self.parameter = Some(p);
            self.range = spec.range.unwrap_or((0.0, 1.0));
        }
        if let Some(plot) = spec.plot {
            self.plotted = Some(plot);
        }
        if let Some(ds_max) = spec.ds_max {
            self.max_step = ds_max;
        }
        self.initial_step = spec.ds;
        self.max_points = spec.max_steps;
    }

    fn start(&mut self, model: &Model) {
        self.error = None;
        let problem = AntimonyProblem::new(model.clone());
        let Some(name) = &self.parameter else { return };
        let Some(active) = problem.parameter_index(name) else {
            self.error = Some(format!("the model has no parameter {name}"));
            return;
        };
        let lambda0 = problem.parameter_values();
        let (lo, hi) = (self.range.0.min(self.range.1), self.range.0.max(self.range.1));
        if !(lo..=hi).contains(&lambda0[active]) {
            self.error = Some(format!("{name} = {} is outside the range; the run starts from the current value", lambda0[active]));
            return;
        }
        let u0 = match problem.find_steady_state(&lambda0) {
            Ok(u) => u,
            Err(e) => {
                self.error = Some(format!("no steady state to start from: {e}"));
                return;
            }
        };
        let defaults = ContinuationOptions::default();
        let options = ContinuationOptions {
            parameter_min: lo,
            parameter_max: hi,
            max_step: self.max_step,
            initial_step: self.initial_step.unwrap_or(defaults.initial_step).min(self.max_step),
            max_points: self.max_points.unwrap_or(defaults.max_points),
            ..defaults
        };
        let model_parameters = model.symbols().count() - model.species.len();
        let mut engines = Vec::new();
        // Both directions from the steady state: features lie on both sides of it.
        for (id, direction) in [(0, 1), (1, -1)] {
            let curve = match EquilibriumCurve::new(AntimonyProblem::new(model.clone()), &lambda0, active, &options) {
                Ok(c) => c,
                Err(e) => {
                    self.error = Some(e.to_string());
                    return;
                }
            };
            let x0 = curve.pack(&u0, lambda0[active]);
            let mut engine = ContinuationEngine::with_branch(curve, options.clone(), Branch::new(id));
            if let Err(e) = engine.initialise(&x0, direction) {
                self.error = Some(format!("the branch could not be started: {e}"));
                return;
            }
            engines.push(engine);
        }
        engines.reverse(); // pop() takes the upward branch first
        let current = engines.pop();
        self.run = Some(Run {
            model: model.clone(),
            parameter_name: name.clone(),
            active,
            model_parameters,
            done: Vec::new(),
            current,
            pending: engines,
            stopped: false,
        });
    }

    pub fn ui(&mut self, ui: &mut egui::Ui, model: &Model) {
        // Everything the model lets us vary: its parameters, then its conserved totals.
        let problem = AntimonyProblem::new(model.clone());
        let names: Vec<String> = (0..problem.parameter_count()).map(|k| problem.parameter_name(k)).collect();
        let values = problem.parameter_values();
        if names.is_empty() {
            ui.label("This model has no parameters to vary.");
            return;
        }
        if self.parameter.as_ref().is_none_or(|p| !names.contains(p)) {
            self.parameter = Some(names[0].clone());
            self.range = default_range(values[0]);
        }
        let species = model.species_names();
        if self.plotted.as_ref().is_none_or(|s| !species.contains(s)) {
            self.plotted = species.first().cloned();
        }

        let running = self.run.as_ref().is_some_and(Run::running);
        ui.horizontal_wrapped(|ui| {
            ui.label("Vary");
            let selected = self.parameter.clone().unwrap_or_default();
            egui::ComboBox::from_id_salt("bif_parameter").selected_text(selected.as_str()).show_ui(ui, |ui| {
                for (k, name) in names.iter().enumerate() {
                    let label = if k < values.len() && name.starts_with("_CSUM") { format!("{name} (conserved total)") } else { name.clone() };
                    if ui.selectable_label(Some(name) == self.parameter.as_ref(), label).clicked() {
                        self.parameter = Some(name.clone());
                        self.range = default_range(values[k]);
                    }
                }
            });
            ui.label("from");
            let speed = (self.range.1 - self.range.0).abs().max(1e-9) * 0.005;
            ui.add(egui::DragValue::new(&mut self.range.0).speed(speed));
            ui.label("to");
            ui.add(egui::DragValue::new(&mut self.range.1).speed(speed));
            ui.label("max step");
            ui.add(egui::DragValue::new(&mut self.max_step).speed(0.005).range(1e-4..=1.0));
            ui.separator();
            if running {
                if ui.button("Stop").clicked()
                    && let Some(run) = &mut self.run
                {
                    run.stopped = true;
                }
            } else if ui.button("Run").on_hover_text("Continue the steady state in both directions").clicked() {
                self.start(model);
            }
            ui.separator();
            ui.label("Plot");
            let selected = self.plotted.clone().unwrap_or_default();
            egui::ComboBox::from_id_salt("bif_species").selected_text(selected.as_str()).show_ui(ui, |ui| {
                for name in &species {
                    ui.selectable_value(&mut self.plotted, Some(name.clone()), name.as_str());
                }
            });
        });

        if let Some(e) = &self.error {
            ui.colored_label(ui.visuals().error_fg_color, e.as_str());
        }
        let Some(run) = &mut self.run else {
            ui.add_space(8.0);
            ui.label("Choose a parameter and a range, then press Run. The branch starts from the steady state at the current slider values.");
            return;
        };

        if run.running() {
            run.advance(STEPS_PER_FRAME);
            ui.ctx().request_repaint();
        }
        let run = &*run;
        let points: usize = run.branches().map(|b| b.points.len()).sum();
        ui.horizontal(|ui| {
            let state = if run.running() {
                "running…".to_owned()
            } else if run.stopped {
                "stopped".to_owned()
            } else {
                run.done.iter().map(|b| b.termination.as_str()).collect::<Vec<_>>().join(", ")
            };
            ui.small(format!("{points} points, {state}"));
            if &run.model != model {
                ui.colored_label(ui.visuals().warn_fg_color, "The model or its values have changed since this run: press Run to update.");
            }
        });

        // The plotted species from a point: the state is the independent
        // species, so dependent ones come back through the conservation laws.
        let conservation = run.model.conservation();
        let plotted = self.plotted.as_ref().and_then(|n| run.model.species_names().iter().position(|s| s == n)).unwrap_or(0);
        let value = |u: &[f64], lambda: &[f64]| -> f64 {
            let totals = &lambda[run.model_parameters.min(lambda.len())..];
            conservation.full_state(u, totals).get(plotted).copied().unwrap_or(f64::NAN)
        };

        let plot_height = (ui.available_height() * 0.7).max(200.0);
        let (stable, unstable) = (Color32::from_rgb(70, 130, 230), Color32::from_rgb(70, 130, 230));
        Plot::new("bifurcation_plot")
            .height(plot_height)
            .legend(Legend::default())
            .x_axis_label(run.parameter_name.as_str())
            .y_axis_label(self.plotted.clone().unwrap_or_default())
            .show(ui, |plot_ui| {
                for branch in run.branches() {
                    // Split the branch where its stability changes; each piece
                    // shares its first point with the last, so the line is unbroken.
                    let mut start = 0;
                    while start + 1 < branch.points.len() {
                        let is_stable = branch.points[start + 1].unstable_dim == 0;
                        let mut end = start + 1;
                        while end + 1 < branch.points.len() && (branch.points[end + 1].unstable_dim == 0) == is_stable {
                            end += 1;
                        }
                        let line: Vec<[f64; 2]> = branch.points[start..=end].iter().map(|p| [p.lambda[run.active], value(&p.u, &p.lambda)]).collect();
                        plot_ui.line(if is_stable {
                            Line::new("stable", line).color(stable).width(2.5)
                        } else {
                            Line::new("unstable", line).color(unstable).width(1.5).style(LineStyle::dashed_loose())
                        });
                        start = end;
                    }
                }
                for b in run.branches().flat_map(|b| &b.bifurcations) {
                    if b.kind == BifurcationKind::NeutralSaddle {
                        continue; // not a bifurcation; listed below
                    }
                    let at = [b.point.lambda[run.active], value(&b.point.u, &b.point.lambda)];
                    let colour = kind_colour(b.kind);
                    plot_ui.points(Points::new(b.kind.as_str(), vec![at]).radius(5.0).color(colour));
                    plot_ui.text(Text::new(b.kind.as_str(), PlotPoint::new(at[0], at[1]), format!("  {}", b.kind.abbreviation())).color(colour).anchor(egui::Align2::LEFT_BOTTOM));
                }
            });

        // The located points, in parameter order.
        let mut found: Vec<_> = run.branches().flat_map(|b| &b.bifurcations).collect();
        found.sort_by(|a, b| a.point.lambda[run.active].total_cmp(&b.point.lambda[run.active]));
        if found.is_empty() {
            return;
        }
        egui::ScrollArea::vertical().id_salt("bif_list").show(ui, |ui| {
            egui::Grid::new("bif_table").striped(true).show(ui, |ui| {
                ui.strong("");
                ui.strong(run.parameter_name.as_str());
                ui.strong("");
                ui.end_row();
                for b in found {
                    ui.colored_label(kind_colour(b.kind), egui::RichText::new(b.kind.abbreviation()).strong());
                    ui.monospace(format!("{:.8}", b.point.lambda[run.active]));
                    let text = match b.kind {
                        BifurcationKind::Hopf => format!("Hopf, ω = {:.6}", b.normal_form.get_or("omega", f64::NAN)),
                        BifurcationKind::Fold => "fold (saddle-node)".to_owned(),
                        BifurcationKind::BranchPoint => "branch point".to_owned(),
                        BifurcationKind::NeutralSaddle => "neutral saddle (not a bifurcation)".to_owned(),
                        k => k.as_str().to_owned(),
                    };
                    ui.label(text);
                    ui.end_row();
                }
            });
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BRUSSELATOR: &str = "
        J0: -> X; A
        J1: X -> Y; B*X
        J2: 2 X + Y -> 3 X; X^2*Y
        J3: X -> ; X
        A = 1; B = 1.2
        X = 1; Y = 1.2";

    /// Every bifurcation example, loaded as the menu loads it, starts from its
    /// steady state inside its own range and runs to the end of both branches.
    #[test]
    fn every_bifurcation_example_runs() {
        use websim_model::model::BIFURCATION_EXAMPLES;
        let mut report = Vec::new();
        let mut failed = false;
        for (name, text) in BIFURCATION_EXAMPLES {
            let model = Model::parse(text).unwrap_or_else(|e| panic!("{name}: {e}"));
            let mut view = BifurcationView::default();
            view.apply_spec(text);
            let spec = RunSpec::parse(text);
            assert!(spec.found && spec.parameter.is_some() && spec.range.is_some(), "{name}: the block gives a parameter and a range");
            assert!(spec.unknown_keys.is_empty(), "{name}: unknown keys {:?}", spec.unknown_keys);
            if let Some(plot) = &spec.plot {
                assert!(model.species_names().contains(plot), "{name}: the block plots {plot}, which is not a species");
            }
            view.start(&model);
            let Some(run) = view.run.as_mut() else {
                failed = true;
                report.push(format!("{name}: {}", view.error.clone().unwrap_or_default()));
                continue;
            };
            while run.running() {
                run.advance(STEPS_PER_FRAME);
            }
            let points: usize = run.branches().map(|b| b.points.len()).sum();
            let found: Vec<String> = run.branches().flat_map(|b| &b.bifurcations).map(|b| format!("{} {:.6}", b.kind.abbreviation(), b.point.lambda[run.active])).collect();
            let ends: Vec<&str> = run.done.iter().map(|b| b.termination.as_str()).collect();
            report.push(format!("{name}: {points} points, {ends:?}, [{}]", found.join(", ")));
        }
        eprintln!("{}", report.join("
"));
        assert!(!failed, "examples that did not start:
{}", report.join("
"));
    }

    /// The view's run finds the Brusselator's Hopf at B = 1 + A² = 2, and
    /// drawing it (headless) does not panic.
    #[test]
    fn brusselator_run_finds_the_hopf_and_draws() {
        let model = Model::parse(BRUSSELATOR).unwrap();
        let mut view = BifurcationView { parameter: Some("B".into()), range: (1.0, 3.0), ..Default::default() };
        view.start(&model);
        assert!(view.error.is_none(), "{:?}", view.error);
        let run = view.run.as_mut().unwrap();
        while run.running() {
            run.advance(STEPS_PER_FRAME);
        }
        assert_eq!(run.done.len(), 2, "both directions ran");
        let hopf: Vec<f64> = run.branches().flat_map(|b| &b.bifurcations).filter(|b| b.kind == BifurcationKind::Hopf).map(|b| b.point.lambda[run.active]).collect();
        assert_eq!(hopf.len(), 1, "one Hopf");
        assert!((hopf[0] - 2.0).abs() < 1e-6, "Hopf at B = {}", hopf[0]);

        let ctx = egui::Context::default();
        for _ in 0..3 {
            let mut output = ctx.run_ui(egui::RawInput::default(), |ui| view.ui(ui, &model));
            output.textures_delta.clear();
        }
    }
}
