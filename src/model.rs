//! An ODE model built from simple statements, plus the expression parser.
//!
//! The model text itself is read by [`crate::antimony`], which turns reactions,
//! rules and assignments into [`Stmt`]s. [`Model::from_statements`] then resolves
//! names, decides what is a species, a parameter or a rule, and checks the result.

use std::cell::Cell;
use std::fmt;

use crate::antimony;
use crate::ode::{SegmentEnd, Solution};
use crate::solvers::{SolverSettings, solve_segment};

/// Example models offered in the UI: (menu name, Antimony text).
pub const EXAMPLES: &[(&str, &str)] = &[
    (
        "Linear pathway",
        "\
// Linear pathway:  $Xo -> S1 -> S2 ->
// '$' marks a boundary species, held constant.
J1: $Xo -> S1; k1*Xo
J2: S1 -> S2;  Vm2*S1/(Km2 + S1)
J3: S2 -> ;    Vm3*S2/(Km3 + S2)

Xo = 10; k1 = 0.5
Vm2 = 8; Km2 = 2
Vm3 = 6; Km3 = 1.5
S1 = 0;  S2 = 0
",
    ),
    (
        "Brusselator",
        "\
// Brusselator: an oscillating chemical reaction.
// Shows stoichiometry (2X, 3X) and boundary species.
J1: $A -> X;          k1*A
J2: $B + X -> Y;      k2*B*X
J3: 2X + Y -> 3X;     k3*X^2*Y
J4: X -> ;            k4*X

A = 1; B = 3
k1 = 1; k2 = 1; k3 = 1; k4 = 1
X = 1; Y = 1
",
    ),
    (
        "Events (relay)",
        "\
// A relay ('bang-bang') controller built from events.
// An event fires when its condition becomes true:
//   at (condition): name = value, name = value
J1: $Xo -> S1; k1*Xo
J2: S1 -> ;    k2*S1

Xo = 10; k1 = 0.5; k2 = 0.5
S1 = 0

off: at (S1 > 4): Xo = 0      // switch the supply off...
on:  at (S1 < 2): Xo = 10     // ...and back on
// At t = 30, kick S1 up. That makes 'off' fire too, at the same moment.
kick: at (time > 30): S1 = S1 + 3
",
    ),
    (
        "Robertson (stiff)",
        "\
// Robertson's reactions: a classic stiff problem.
// The rate constants differ by a factor of 10^9, so
// RK4 needs tiny steps; BDF and ESDIRK34 handle it easily.
// (B stays tiny: untick A and C to see it.)
J1: A -> B;          k1*A
J2: B + C -> A + C;  k2*B*C
J3: 2B -> B + C;     k3*B^2

k1 = 0.04; k2 = 1e4; k3 = 3e7
A = 1; B = 0; C = 0
",
    ),
    (
        "Lotka–Volterra",
        "\
// Lotka–Volterra predator–prey model, written with
// rate rules:  name' = expression
prey' = alpha*prey - beta*prey*pred
pred' = delta*prey*pred - gamma*pred

alpha = 1.1; beta = 0.4
delta = 0.1; gamma = 0.4
prey = 10;   pred = 10
",
    ),
    (
        "Lorenz attractor",
        "\
// Lorenz system: chaotic for these parameters.
// Try plotting z against x in the phase plane.
x' = sigma*(y - x)
y' = x*(rho - z) - y
z' = x*y - beta*z

sigma = 10; rho = 28; beta = 8/3
x = 1; y = 1; z = 1
",
    ),
    (
        "Forced oscillator",
        "\
// Damped spring pushed by a periodic force.
// ':=' defines a rule, recomputed at every step;
// 'time' is the simulation time.
force := F*sin(w*time)
x' = v
v' = -k*x - c*v + force

k = 1; c = 0.2
F = 0.5; w = 0.8
x = 1; v = 0
",
    ),
];

/// One statement produced by the Antimony front end.
#[derive(Debug)]
pub struct Stmt {
    pub line: usize,
    pub name: String,
    pub kind: StmtKind,
    pub rhs: String,
}

#[derive(Debug, PartialEq)]
pub enum StmtKind {
    /// `name' = rhs`: the rate of change of a species.
    Rate,
    /// `name = rhs`: a value set once, at the start.
    Assign,
    /// `name := rhs`: a rule, recomputed at every step.
    Rule,
}

/// An event as written: `name: at (trigger): target = value, ...`
#[derive(Debug)]
pub struct EventStmt {
    pub line: usize,
    pub name: Option<String>,
    pub trigger: String,
    /// (target name, value expression)
    pub assignments: Vec<(String, String)>,
}

/// An event, ready to simulate.
#[derive(Clone, Debug, PartialEq)]
struct Event {
    name: String,
    trigger: Expr,
    assignments: Vec<(Target, Expr)>,
}

/// What an event assignment changes.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Target {
    Species(usize),
    Var(usize),
}

/// Stop if events keep setting each other off at one moment, or fire this often in a run.
const MAX_EVENT_ROUNDS: usize = 100;
const MAX_EVENTS: usize = 10_000;

/// A named value the user can adjust: a parameter or a species' initial value.
#[derive(Clone, Debug, PartialEq)]
pub struct Symbol {
    pub name: String,
    pub value: f64,
    /// The value written in the model text.
    pub default: f64,
}

impl Symbol {
    fn new(name: &str, value: f64) -> Self {
        Self {
            name: name.to_owned(),
            value,
            default: value,
        }
    }
}

/// A name that isn't a species: a parameter or a rule.
#[derive(Clone, Debug, PartialEq)]
struct Var {
    symbol: Symbol,
    /// `None` for a parameter, otherwise the expression recomputed at each step.
    rule: Option<Expr>,
    line: usize,
}

/// A parsed model, ready to simulate.
#[derive(Clone, Debug, PartialEq)]
pub struct Model {
    /// Species in order of appearance; `value` is the initial value.
    pub species: Vec<Symbol>,
    vars: Vec<Var>,
    /// Indices into `vars` of the rules, sorted so each comes after the rules it uses.
    rule_order: Vec<usize>,
    /// `rates[i]` is d(species[i])/dt.
    rates: Vec<Expr>,
    events: Vec<Event>,
}

#[derive(Debug)]
pub struct ParseError {
    /// 1-based line number, or 0 if the error isn't about one line.
    pub line: usize,
    pub message: String,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.line == 0 {
            write!(f, "{}", self.message)
        } else {
            write!(f, "Line {}: {}", self.line, self.message)
        }
    }
}

impl Model {
    /// Parse a model written in (a basic subset of) Antimony.
    pub fn parse(text: &str) -> Result<Model, ParseError> {
        let parsed = antimony::statements(text)?;
        Model::from_statements(&parsed.stmts, &parsed.events)
    }

    pub fn from_statements(stmts: &[Stmt], events: &[EventStmt]) -> Result<Model, ParseError> {
        for s in stmts {
            if !is_identifier(&s.name) {
                return Err(error(s.line, format!("'{}' is not a valid name", s.name)));
            }
            if is_reserved(&s.name) {
                return Err(error(s.line, format!("'{}' is a reserved name", s.name)));
            }
        }

        // Species are the names with a rate equation. Every other assigned
        // name is a variable: a parameter or a rule.
        let mut species: Vec<&str> = Vec::new();
        for s in stmts.iter().filter(|s| s.kind == StmtKind::Rate) {
            if species.contains(&s.name.as_str()) {
                return Err(error(s.line, format!("'{}' has more than one rate rule", s.name)));
            }
            species.push(&s.name);
        }
        if species.is_empty() {
            return Err(error(
                0,
                "No reactions or rate rules yet. Add one, e.g.  J1: S1 -> S2; k1*S1".to_owned(),
            ));
        }
        let mut var_names: Vec<&str> = Vec::new();
        for s in stmts.iter().filter(|s| s.kind != StmtKind::Rate) {
            if species.contains(&s.name.as_str()) {
                continue;
            }
            if var_names.contains(&s.name.as_str()) {
                return Err(error(s.line, format!("'{}' is assigned more than once", s.name)));
            }
            var_names.push(&s.name);
        }

        let lookup = |name: &str| -> Option<Expr> {
            if let Some(i) = species.iter().position(|s| *s == name) {
                return Some(Expr::Species(i));
            }
            if let Some(i) = var_names.iter().position(|v| *v == name) {
                return Some(Expr::Var(i));
            }
            match name {
                "time" => Some(Expr::Time),
                "pi" => Some(Expr::Num(std::f64::consts::PI)),
                _ => None,
            }
        };

        let mut model = Model {
            species: species.iter().map(|name| Symbol::new(name, 0.0)).collect(),
            vars: Vec::new(),
            rule_order: Vec::new(),
            rates: vec![Expr::Num(0.0); species.len()],
            events: Vec::new(),
        };
        let mut vars: Vec<Option<Var>> = vec![None; var_names.len()];
        let mut initial_value_set = vec![false; species.len()];

        for s in stmts {
            let err = |message: String| error(s.line, message);
            let expr = parse_expr(&s.rhs, &lookup).map_err(err)?;
            let species_index = species.iter().position(|n| *n == s.name);

            match (&s.kind, species_index) {
                (StmtKind::Rate, Some(i)) => model.rates[i] = expr,
                (StmtKind::Rule, Some(_)) => {
                    return Err(err(format!(
                        "'{}' is a species, so it can't also have a ':=' rule",
                        s.name
                    )));
                }
                (StmtKind::Assign, Some(i)) => {
                    if initial_value_set[i] {
                        return Err(err(format!("the initial value of '{}' is set twice", s.name)));
                    }
                    if expr.uses_names() {
                        return Err(err(format!(
                            "the initial value of '{}' must be a number (this parser doesn't \
                             support initial assignments to species yet)",
                            s.name
                        )));
                    }
                    initial_value_set[i] = true;
                    model.species[i] = Symbol::new(&s.name, expr.constant_value());
                }
                (StmtKind::Rate, None) => unreachable!("every rate statement defines a species"),
                (kind, None) => {
                    let j = var_names.iter().position(|n| *n == s.name).unwrap();
                    let var = if *kind == StmtKind::Rule || expr.uses_names() {
                        // In Antimony, `=` sets a value once at the start. When it only
                        // uses parameters, recomputing it every step gives the same result.
                        if *kind == StmtKind::Assign && expr.uses_species_or_time() {
                            return Err(err(format!(
                                "'=' sets '{}' once, at the start, which this parser only \
                                 supports for parameters. Use ':=' to recompute it every step.",
                                s.name
                            )));
                        }
                        Var { symbol: Symbol::new(&s.name, 0.0), rule: Some(expr), line: s.line }
                    } else {
                        Var {
                            symbol: Symbol::new(&s.name, expr.constant_value()),
                            rule: None,
                            line: s.line,
                        }
                    };
                    vars[j] = Some(var);
                }
            }
        }
        model.vars = vars
            .into_iter()
            .map(|v| v.expect("every variable has an assignment"))
            .collect();
        model.rule_order = rule_order(&model.vars)?;

        for (k, e) in events.iter().enumerate() {
            let err = |message: String| error(e.line, message);
            let trigger = parse_expr(&e.trigger, &lookup).map_err(err)?;
            let mut assignments = Vec::new();
            for (target, value) in &e.assignments {
                let target = match lookup(target) {
                    Some(Expr::Species(i)) => Target::Species(i),
                    Some(Expr::Var(j)) if model.vars[j].rule.is_none() => Target::Var(j),
                    Some(Expr::Var(_)) => {
                        return Err(err(format!(
                            "'{target}' is calculated from other values, so an event can't set it"
                        )));
                    }
                    _ => {
                        return Err(err(format!(
                            "an event can only set species and parameters, and '{target}' is neither"
                        )));
                    }
                };
                assignments.push((target, parse_expr(value, &lookup).map_err(err)?));
            }
            let name = e.name.clone().unwrap_or_else(|| format!("event {}", k + 1));
            model.events.push(Event { name, trigger, assignments });
        }
        Ok(model)
    }

    /// The parameters, which the UI shows as sliders.
    pub fn params_mut(&mut self) -> impl Iterator<Item = &mut Symbol> {
        self.vars
            .iter_mut()
            .filter(|v| v.rule.is_none())
            .map(|v| &mut v.symbol)
    }

    /// Every adjustable value: species' initial values, then parameters.
    pub fn symbols(&self) -> impl Iterator<Item = &Symbol> {
        let params = self.vars.iter().filter(|v| v.rule.is_none()).map(|v| &v.symbol);
        self.species.iter().chain(params)
    }

    fn symbols_mut(&mut self) -> impl Iterator<Item = &mut Symbol> {
        let params = self
            .vars
            .iter_mut()
            .filter(|v| v.rule.is_none())
            .map(|v| &mut v.symbol);
        self.species.iter_mut().chain(params)
    }

    /// Put every slider back to the value written in the model text.
    pub fn reset_values(&mut self) {
        for s in self.symbols_mut() {
            s.value = s.default;
        }
    }

    /// After the text is edited, keep any slider adjustments the user made
    /// to values whose name and written value haven't changed.
    pub fn keep_values_from(&mut self, old: &Model) {
        for s in self.symbols_mut() {
            if let Some(o) = old.symbols().find(|o| o.name == s.name && o.default == s.default) {
                s.value = o.value;
            }
        }
    }

    /// d(species)/dt at time `t` and state `y`, with parameter values `params`.
    fn rates_into(&self, t: f64, y: &[f64], params: &[f64], out: &mut [f64]) {
        let vars = self.eval_vars(t, y, params);
        let ctx = Ctx { t, y, vars: &vars };
        for (o, rate) in out.iter_mut().zip(&self.rates) {
            *o = rate.eval(&ctx);
        }
    }

    /// Simulate from `t = 0` to `t_end` with the chosen solver.
    ///
    /// Events follow Antimony's and libRoadRunner's defaults:
    /// * an event fires when its trigger changes from false to true, and a trigger
    ///   that is already true at the start does not fire (`t0 = true`);
    /// * all of an event's assignments are computed from the values at the moment
    ///   it fires, then applied together (`fromTrigger = true`). When several events
    ///   fire at once, they are all computed first and applied in document order;
    /// * if the assignments make another trigger true, that event fires at the same time.
    ///
    /// The run is split into segments at events: the solver integrates until a trigger
    /// becomes true, the events are applied, and the solver restarts from the new state.
    pub fn simulate(&self, t_end: f64, settings: &SolverSettings) -> Solution {
        let mut params: Vec<f64> = self.vars.iter().map(|v| v.symbol.value).collect();
        let mut y: Vec<f64> = self.species.iter().map(|s| s.value).collect();
        let mut t = 0.0;
        let mut was_true: Vec<bool> = self.triggers(t, &y, &params);

        let mut solution = Solution {
            t: vec![t],
            y: vec![y.clone()],
            param_changes: vec![(0, params.clone())],
            ..Default::default()
        };
        let evaluations = Cell::new(0);

        loop {
            let end = {
                let rates = |t: f64, y: &[f64], out: &mut [f64]| {
                    evaluations.set(evaluations.get() + 1);
                    self.rates_into(t, y, &params, out);
                };
                let mut check_events = |t0: f64, t1: f64, y_at: &dyn Fn(f64) -> Vec<f64>| {
                    self.find_event(t0, t1, y_at, &params, &mut was_true)
                };
                solve_segment(settings, &rates, &y, t, t_end, &mut solution, &mut check_events)
            };

            match end {
                SegmentEnd::Done => break,
                SegmentEnd::Failed(reason) => {
                    solution.stopped_early = Some(reason);
                    break;
                }
                SegmentEnd::Event { t: t_event, y: y_event } => {
                    t = t_event;
                    y = y_event;
                    solution.t.push(t); // the state just before the event...
                    solution.y.push(y.clone());
                    if let Err(reason) = self.fire_events(t, &mut y, &mut params, &mut was_true, &mut solution) {
                        solution.stopped_early = Some(reason);
                        break;
                    }
                    solution.t.push(t); // ...and just after
                    solution.y.push(y.clone());
                    solution.param_changes.push((solution.t.len() - 1, params.clone()));
                    if solution.events.len() > MAX_EVENTS {
                        solution.stopped_early =
                            Some(format!("Stopped at t = {t:.4} after {MAX_EVENTS} events."));
                        break;
                    }
                }
            }
        }
        solution.rate_evaluations = evaluations.get();
        solution
    }

    /// The value of every event trigger.
    fn triggers(&self, t: f64, y: &[f64], params: &[f64]) -> Vec<bool> {
        if self.events.is_empty() {
            return Vec::new();
        }
        let vars = self.eval_vars(t, y, params);
        let ctx = Ctx { t, y, vars: &vars };
        self.events.iter().map(|e| e.trigger.eval(&ctx) != 0.0).collect()
    }

    /// Called after each solver step from `t0` to `t1`. If a trigger that was false
    /// at `t0` is true at `t1`, find when it became true by bisection, and return the
    /// earliest such time. Otherwise remember the trigger values at `t1`.
    fn find_event(
        &self,
        t0: f64,
        t1: f64,
        y_at: &dyn Fn(f64) -> Vec<f64>,
        params: &[f64],
        was_true: &mut [bool],
    ) -> Option<f64> {
        if self.events.is_empty() {
            return None;
        }
        let now = self.triggers(t1, &y_at(t1), params);
        let mut earliest: Option<f64> = None;
        for k in 0..self.events.len() {
            if was_true[k] || !now[k] {
                continue;
            }
            // False at `low`, true at `high`: narrow the gap down.
            let (mut low, mut high) = (t0, t1);
            while high - low > 1e-12 * (1.0 + high.abs()) {
                let mid = 0.5 * (low + high);
                if self.triggers(mid, &y_at(mid), params)[k] {
                    high = mid;
                } else {
                    low = mid;
                }
            }
            earliest = Some(earliest.map_or(high, |e: f64| e.min(high)));
        }
        if earliest.is_none() {
            was_true.copy_from_slice(&now);
        }
        earliest
    }

    /// Fire every event whose trigger has just become true, then any events that
    /// those assignments set off in turn.
    fn fire_events(
        &self,
        t: f64,
        y: &mut [f64],
        params: &mut [f64],
        was_true: &mut [bool],
        solution: &mut Solution,
    ) -> Result<(), String> {
        for _ in 0..MAX_EVENT_ROUNDS {
            let now = self.triggers(t, y, params);
            let firing: Vec<usize> = (0..now.len()).filter(|&k| now[k] && !was_true[k]).collect();
            was_true.copy_from_slice(&now);
            if firing.is_empty() {
                return Ok(());
            }

            // Compute every assignment from the values at this moment, then apply them.
            let updates: Vec<(Target, f64)> = {
                let vars = self.eval_vars(t, y, params);
                let ctx = Ctx { t, y, vars: &vars };
                firing
                    .iter()
                    .flat_map(|&k| &self.events[k].assignments)
                    .map(|(target, value)| (*target, value.eval(&ctx)))
                    .collect()
            };
            for (target, value) in updates {
                match target {
                    Target::Species(i) => y[i] = value,
                    Target::Var(j) => params[j] = value,
                }
            }
            solution.events.extend(firing.iter().map(|&k| (t, k)));
        }
        Err(format!(
            "Stopped at t = {t:.4}: events kept setting each other off ({MAX_EVENT_ROUNDS} rounds)."
        ))
    }

    /// Simulate, then also work out every rule (including reaction rates)
    /// at each time point, so anything can be plotted.
    pub fn run(&self, t_end: f64, settings: &SolverSettings) -> Results {
        let solution = self.simulate(t_end, settings);
        let rules: Vec<usize> = (0..self.vars.len())
            .filter(|&j| self.vars[j].rule.is_some())
            .collect();

        let mut names: Vec<String> = self.species.iter().map(|s| s.name.clone()).collect();
        names.extend(rules.iter().map(|&j| self.vars[j].symbol.name.clone()));

        // Rules use the parameter values in force at each point, which events may change.
        let mut columns = vec![Vec::with_capacity(solution.t.len()); names.len()];
        let mut changes = solution.param_changes.iter().peekable();
        let mut params: &[f64] = &[];
        for (i, (&t, y)) in solution.t.iter().zip(&solution.y).enumerate() {
            while let Some((_, p)) = changes.next_if(|(from, _)| *from <= i) {
                params = p;
            }
            let vars = self.eval_vars(t, y, params);
            let values = y.iter().copied().chain(rules.iter().map(|&j| vars[j]));
            for (column, value) in columns.iter_mut().zip(values) {
                column.push(value);
            }
        }

        Results {
            names,
            species_count: self.species.len(),
            t: solution.t,
            columns,
            events: solution
                .events
                .iter()
                .map(|&(t, k)| (t, self.events[k].name.clone()))
                .collect(),
            stopped_early: solution.stopped_early,
            steps: solution.steps,
            rate_evaluations: solution.rate_evaluations,
        }
    }

    /// The values of all variables at time `t` and state `y`:
    /// parameters from `params`, rules computed in dependency order.
    fn eval_vars(&self, t: f64, y: &[f64], params: &[f64]) -> Vec<f64> {
        let mut vars = params.to_vec();
        for &j in &self.rule_order {
            let rule = self.vars[j].rule.as_ref().unwrap();
            let value = rule.eval(&Ctx { t, y, vars: &vars });
            vars[j] = value;
        }
        vars
    }
}

/// Everything that can be plotted from a simulation, one column per quantity:
/// first the species, then the rules (which include reaction rates).
#[derive(Default)]
pub struct Results {
    pub names: Vec<String>,
    pub species_count: usize,
    pub t: Vec<f64>,
    pub columns: Vec<Vec<f64>>,
    /// Events that fired: (time, event name).
    pub events: Vec<(f64, String)>,
    /// Why the solver stopped before the end time, if it did.
    pub stopped_early: Option<String>,
    pub steps: usize,
    pub rate_evaluations: usize,
}

impl Results {
    /// One quantity over time as `[t, value]` points, ready for plotting.
    pub fn series(&self, index: usize) -> Vec<[f64; 2]> {
        self.t.iter().zip(&self.columns[index]).map(|(t, v)| [*t, *v]).collect()
    }
}

fn error(line: usize, message: String) -> ParseError {
    ParseError { line, message }
}

/// Sort the rules so that each one comes after every rule it uses,
/// with a depth-first search. Reports circular definitions.
fn rule_order(vars: &[Var]) -> Result<Vec<usize>, ParseError> {
    #[derive(Clone, Copy, PartialEq)]
    enum State {
        New,
        InProgress,
        Done,
    }

    fn visit(j: usize, vars: &[Var], state: &mut [State], order: &mut Vec<usize>) -> Result<(), ParseError> {
        match state[j] {
            State::Done => return Ok(()),
            State::InProgress => {
                let message = format!("'{}' is part of a circular definition", vars[j].symbol.name);
                return Err(error(vars[j].line, message));
            }
            State::New => {}
        }
        state[j] = State::InProgress;
        if let Some(rule) = &vars[j].rule {
            let mut uses = Vec::new();
            rule.visit(&mut |e| {
                if let Expr::Var(k) = e {
                    if vars[*k].rule.is_some() {
                        uses.push(*k);
                    }
                }
            });
            for k in uses {
                visit(k, vars, state, order)?;
            }
            order.push(j);
        }
        state[j] = State::Done;
        Ok(())
    }

    let mut state = vec![State::New; vars.len()];
    let mut order = Vec::new();
    for j in 0..vars.len() {
        visit(j, vars, &mut state, &mut order)?;
    }
    Ok(order)
}

pub fn is_identifier(s: &str) -> bool {
    let mut chars = s.chars();
    chars.next().is_some_and(|c| c.is_alphabetic() || c == '_')
        && chars.all(|c| c.is_alphanumeric() || c == '_')
}

fn is_reserved(name: &str) -> bool {
    matches!(name, "time" | "pi" | "log" | "and" | "or" | "not") || Func::from_name(name).is_some()
}

// ---------------------------------------------------------------------------
// Expressions

#[derive(Clone, Copy, Debug, PartialEq)]
enum Func {
    Exp,
    Ln,
    Log10,
    Sqrt,
    Abs,
    Sin,
    Cos,
    Tan,
    Floor,
    Ceil,
    Pow,
    Min,
    Max,
}

impl Func {
    fn from_name(name: &str) -> Option<Func> {
        Some(match name {
            "exp" => Func::Exp,
            "ln" => Func::Ln,
            "log10" => Func::Log10,
            "sqrt" => Func::Sqrt,
            "abs" => Func::Abs,
            "sin" => Func::Sin,
            "cos" => Func::Cos,
            "tan" => Func::Tan,
            "floor" => Func::Floor,
            "ceil" => Func::Ceil,
            "pow" => Func::Pow,
            "min" => Func::Min,
            "max" => Func::Max,
            _ => return None,
        })
    }

    fn arity(self) -> usize {
        match self {
            Func::Pow | Func::Min | Func::Max => 2,
            _ => 1,
        }
    }
}

/// A parsed expression. Names are resolved to indices when parsing,
/// so evaluation never has to look anything up by name.
#[derive(Clone, Debug, PartialEq)]
enum Expr {
    Num(f64),
    Time,
    Species(usize),
    Var(usize),
    Neg(Box<Expr>),
    /// Logical not: 1 if the operand is 0, else 0.
    Not(Box<Expr>),
    /// A binary operator: one of `+ - * / ^`, a comparison `< ≤ > ≥ = ≠`
    /// (written `< <= > >= == !=`), or `&` / `|` for `&&` / `||`.
    /// Comparisons and logic give 1 for true and 0 for false.
    Bin(char, Box<Expr>, Box<Expr>),
    Call(Func, Vec<Expr>),
}

/// The values an expression can refer to while it is evaluated.
struct Ctx<'a> {
    t: f64,
    y: &'a [f64],
    vars: &'a [f64],
}

impl Expr {
    fn eval(&self, c: &Ctx) -> f64 {
        match self {
            Expr::Num(v) => *v,
            Expr::Time => c.t,
            Expr::Species(i) => c.y[*i],
            Expr::Var(i) => c.vars[*i],
            Expr::Neg(e) => -e.eval(c),
            Expr::Not(e) => truth(e.eval(c) == 0.0),
            Expr::Bin(op, a, b) => {
                let (a, b) = (a.eval(c), b.eval(c));
                match op {
                    '+' => a + b,
                    '-' => a - b,
                    '*' => a * b,
                    '/' => a / b,
                    '^' => a.powf(b),
                    '<' => truth(a < b),
                    '≤' => truth(a <= b),
                    '>' => truth(a > b),
                    '≥' => truth(a >= b),
                    '=' => truth(a == b),
                    '≠' => truth(a != b),
                    '&' => truth(a != 0.0 && b != 0.0),
                    '|' => truth(a != 0.0 || b != 0.0),
                    _ => unreachable!("unknown operator {op}"),
                }
            }
            Expr::Call(func, args) => {
                let x = args[0].eval(c);
                match func {
                    Func::Exp => x.exp(),
                    Func::Ln => x.ln(),
                    Func::Log10 => x.log10(),
                    Func::Sqrt => x.sqrt(),
                    Func::Abs => x.abs(),
                    Func::Sin => x.sin(),
                    Func::Cos => x.cos(),
                    Func::Tan => x.tan(),
                    Func::Floor => x.floor(),
                    Func::Ceil => x.ceil(),
                    Func::Pow => x.powf(args[1].eval(c)),
                    Func::Min => x.min(args[1].eval(c)),
                    Func::Max => x.max(args[1].eval(c)),
                }
            }
        }
    }

    /// Call `f` on this node and every node below it.
    fn visit(&self, f: &mut impl FnMut(&Expr)) {
        f(self);
        match self {
            Expr::Neg(e) | Expr::Not(e) => e.visit(f),
            Expr::Bin(_, a, b) => {
                a.visit(f);
                b.visit(f);
            }
            Expr::Call(_, args) => args.iter().for_each(|a| a.visit(f)),
            _ => {}
        }
    }

    /// True if the expression refers to time, a species or a variable.
    fn uses_names(&self) -> bool {
        let mut found = false;
        self.visit(&mut |e| {
            if matches!(e, Expr::Time | Expr::Species(_) | Expr::Var(_)) {
                found = true;
            }
        });
        found
    }

    fn uses_species_or_time(&self) -> bool {
        let mut found = false;
        self.visit(&mut |e| {
            if matches!(e, Expr::Time | Expr::Species(_)) {
                found = true;
            }
        });
        found
    }

    /// The value of an expression that uses no names.
    fn constant_value(&self) -> f64 {
        self.eval(&Ctx { t: 0.0, y: &[], vars: &[] })
    }
}

/// How comparisons and logic represent true and false.
fn truth(b: bool) -> f64 {
    if b { 1.0 } else { 0.0 }
}

// ---------------------------------------------------------------------------
// Tokenizer and recursive-descent parser for expressions

#[derive(Clone, Debug, PartialEq)]
enum Token {
    Num(f64),
    Name(String),
    /// One of `+ - * / ^ ( ) , < > !`, or a two-character operator stored as one
    /// character: `≤ ≥ = ≠ & |` for `<= >= == != && ||`.
    Sym(char),
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Token::Num(v) => write!(f, "'{v}'"),
            Token::Name(n) => write!(f, "'{n}'"),
            Token::Sym(c) => {
                let written = TWO_CHAR_OPERATORS
                    .iter()
                    .find(|(_, sym)| sym == c)
                    .map_or(c.to_string(), |(text, _)| text.to_string());
                write!(f, "'{written}'")
            }
        }
    }
}

/// Operators written with two characters, and the single character used for each in [`Token::Sym`].
const TWO_CHAR_OPERATORS: [(&str, char); 6] = [
    ("<=", '≤'),
    (">=", '≥'),
    ("==", '='),
    ("!=", '≠'),
    ("&&", '&'),
    ("||", '|'),
];

fn tokenize(text: &str) -> Result<Vec<Token>, String> {
    let chars: Vec<char> = text.chars().collect();
    let mut tokens = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        let next = chars.get(i + 1).copied();
        if c.is_whitespace() {
            i += 1;
        } else if c.is_ascii_digit() || (c == '.' && next.is_some_and(|d| d.is_ascii_digit())) {
            let start = i;
            while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                i += 1;
            }
            // Optional exponent, e.g. 1.5e-3
            if i < chars.len() && (chars[i] == 'e' || chars[i] == 'E') {
                let mut j = i + 1;
                if j < chars.len() && (chars[j] == '+' || chars[j] == '-') {
                    j += 1;
                }
                if j < chars.len() && chars[j].is_ascii_digit() {
                    i = j;
                    while i < chars.len() && chars[i].is_ascii_digit() {
                        i += 1;
                    }
                }
            }
            let number: String = chars[start..i].iter().collect();
            let value = number.parse().map_err(|_| format!("'{number}' is not a valid number"))?;
            tokens.push(Token::Num(value));
        } else if c.is_alphabetic() || c == '_' {
            let start = i;
            while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            let name: String = chars[start..i].iter().collect();
            tokens.push(match name.as_str() {
                "and" => Token::Sym('&'),
                "or" => Token::Sym('|'),
                "not" => Token::Sym('!'),
                _ => Token::Name(name),
            });
        } else if c == '*' && next == Some('*') {
            tokens.push(Token::Sym('^'));
            i += 2;
        } else if let Some((_, sym)) = TWO_CHAR_OPERATORS
            .iter()
            .find(|(text, _)| next.is_some_and(|n| text.starts_with(c) && text.ends_with(n)))
        {
            tokens.push(Token::Sym(*sym));
            i += 2;
        } else if "+-*/^(),<>!".contains(c) {
            tokens.push(Token::Sym(c));
            i += 1;
        } else {
            return Err(format!("unexpected character '{c}'"));
        }
    }
    Ok(tokens)
}

/// Parse one expression. `lookup` turns a name into the expression it stands for.
fn parse_expr(text: &str, lookup: &dyn Fn(&str) -> Option<Expr>) -> Result<Expr, String> {
    let mut parser = Parser {
        tokens: tokenize(text)?,
        pos: 0,
        lookup,
    };
    if parser.tokens.is_empty() {
        return Err("missing expression".to_owned());
    }
    let expr = parser.expr()?;
    match parser.peek() {
        Some(token) => Err(format!("unexpected {token}")),
        None => Ok(expr),
    }
}

/// Grammar, from lowest to highest precedence:
///
/// ```text
/// expr  := and ('||' and)*
/// and   := cmp ('&&' cmp)*
/// cmp   := sum (('<' | '<=' | '>' | '>=' | '==' | '!=') sum)?
/// sum   := term (('+' | '-') term)*
/// term  := unary (('*' | '/') unary)*
/// unary := ('-' | '+' | '!') unary | power
/// power := atom ('^' unary)?            right-associative: 2^3^2 = 2^(3^2)
/// atom  := number | name | func '(' expr (',' expr)* ')' | '(' expr ')'
/// ```
struct Parser<'a> {
    tokens: Vec<Token>,
    pos: usize,
    lookup: &'a dyn Fn(&str) -> Option<Expr>,
}

impl Parser<'_> {
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    /// Consume the symbol `c` if it comes next.
    fn eat(&mut self, c: char) -> bool {
        if self.peek() == Some(&Token::Sym(c)) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn expect(&mut self, c: char) -> Result<(), String> {
        if self.eat(c) {
            return Ok(());
        }
        Err(match self.peek() {
            Some(token) => format!("expected '{c}' but found {token}"),
            None => format!("expected '{c}' at the end"),
        })
    }

    fn expr(&mut self) -> Result<Expr, String> {
        let mut lhs = self.and()?;
        while self.eat('|') {
            lhs = Expr::Bin('|', Box::new(lhs), Box::new(self.and()?));
        }
        Ok(lhs)
    }

    fn and(&mut self) -> Result<Expr, String> {
        let mut lhs = self.comparison()?;
        while self.eat('&') {
            lhs = Expr::Bin('&', Box::new(lhs), Box::new(self.comparison()?));
        }
        Ok(lhs)
    }

    fn comparison(&mut self) -> Result<Expr, String> {
        let lhs = self.sum()?;
        for op in ['<', '≤', '>', '≥', '=', '≠'] {
            if self.eat(op) {
                return Ok(Expr::Bin(op, Box::new(lhs), Box::new(self.sum()?)));
            }
        }
        Ok(lhs)
    }

    fn sum(&mut self) -> Result<Expr, String> {
        let mut lhs = self.term()?;
        loop {
            let op = if self.eat('+') {
                '+'
            } else if self.eat('-') {
                '-'
            } else {
                return Ok(lhs);
            };
            lhs = Expr::Bin(op, Box::new(lhs), Box::new(self.term()?));
        }
    }

    fn term(&mut self) -> Result<Expr, String> {
        let mut lhs = self.unary()?;
        loop {
            let op = if self.eat('*') {
                '*'
            } else if self.eat('/') {
                '/'
            } else {
                return Ok(lhs);
            };
            lhs = Expr::Bin(op, Box::new(lhs), Box::new(self.unary()?));
        }
    }

    fn unary(&mut self) -> Result<Expr, String> {
        if self.eat('-') {
            Ok(Expr::Neg(Box::new(self.unary()?)))
        } else if self.eat('!') {
            Ok(Expr::Not(Box::new(self.unary()?)))
        } else if self.eat('+') {
            self.unary()
        } else {
            self.power()
        }
    }

    fn power(&mut self) -> Result<Expr, String> {
        let base = self.atom()?;
        if self.eat('^') {
            Ok(Expr::Bin('^', Box::new(base), Box::new(self.unary()?)))
        } else {
            Ok(base)
        }
    }

    fn atom(&mut self) -> Result<Expr, String> {
        let token = self.peek().cloned().ok_or("the expression ends too early")?;
        self.pos += 1;
        match token {
            Token::Num(v) => Ok(Expr::Num(v)),
            Token::Sym('(') => {
                let e = self.expr()?;
                self.expect(')')?;
                Ok(e)
            }
            Token::Name(name) if name == "log" => {
                Err("log() is ambiguous: use ln() for natural log or log10()".to_owned())
            }
            Token::Name(name) => match Func::from_name(&name) {
                Some(func) => {
                    self.expect('(')?;
                    let mut args = vec![self.expr()?];
                    while self.eat(',') {
                        args.push(self.expr()?);
                    }
                    self.expect(')')?;
                    if args.len() != func.arity() {
                        return Err(format!("{name}() takes {} argument(s)", func.arity()));
                    }
                    Ok(Expr::Call(func, args))
                }
                None => (self.lookup)(&name).ok_or_else(|| format!("unknown name '{name}'")),
            },
            other => Err(format!("unexpected {other}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solvers::Method;

    fn calc(text: &str) -> f64 {
        parse_expr(text, &|_| None).unwrap().constant_value()
    }

    fn parse_error(text: &str) -> String {
        Model::parse(text).unwrap_err().to_string()
    }

    /// Default settings for each of the four solvers.
    fn every_solver() -> impl Iterator<Item = SolverSettings> {
        Method::ALL.into_iter().map(|method| SolverSettings { method, ..Default::default() })
    }

    #[test]
    fn comparison_and_logic_operators() {
        assert_eq!(calc("1 < 2"), 1.0);
        assert_eq!(calc("2 <= 1"), 0.0);
        assert_eq!(calc("1 + 1 > 1"), 1.0); // arithmetic binds tighter than comparison
        assert_eq!(calc("3 >= 3 && 2 != 2"), 0.0);
        assert_eq!(calc("1 == 2 || 2 == 2"), 1.0);
        assert_eq!(calc("!0 and not 0"), 1.0);
        assert_eq!(calc("1 > 2 or 1"), 1.0);
    }

    /// x rises at rate k until t = 5, when an event reverses k: so x peaks at 5 and x(10) = 0.
    #[test]
    fn time_event_fires_at_the_right_moment() {
        let model = Model::parse("x' = k\nk = 1\nx = 0\nflip: at (time > 5): k = -1").unwrap();
        for settings in every_solver() {
            let res = model.run(10.0, &settings);
            let method = settings.method;
            assert_eq!(res.events.len(), 1, "{method:?}");
            let (t, name) = &res.events[0];
            assert!((t - 5.0).abs() < 1e-9, "{method:?}: fired at {t}");
            assert_eq!(name, "flip");
            let x = &res.columns[0];
            assert!(x.last().unwrap().abs() < 1e-6, "{method:?}: x(10) = {}", x.last().unwrap());
            let peak = x.iter().copied().fold(f64::MIN, f64::max);
            assert!((peak - 5.0).abs() < 1e-6, "{method:?}: peak = {peak}");
        }
    }

    /// x decays as e^-t and is reset to 1 whenever it drops below 0.5, i.e. every ln 2.
    /// Event times add up, so solver error accumulates: use tight tolerances.
    #[test]
    fn state_event_fires_repeatedly() {
        let model = Model::parse("x' = -x\nx = 1\nat (x < 0.5): x = 1").unwrap();
        for settings in every_solver() {
            let settings = SolverSettings { rtol: 1e-9, atol: 1e-12, ..settings };
            let res = model.run(10.0, &settings);
            let method = settings.method;
            assert_eq!(res.events.len(), 14, "{method:?}"); // 10 / ln 2 = 14.4
            for (n, (t, _)) in res.events.iter().enumerate() {
                let expected = (n + 1) as f64 * 2f64.ln();
                assert!((t - expected).abs() < 1e-5, "{method:?}: event {n} at {t}, expected {expected}");
            }
            assert_eq!(res.events[0].1, "event 1"); // unnamed events are numbered
        }
    }

    /// Antimony's default `t0 = true`: a trigger already true at the start doesn't fire.
    #[test]
    fn trigger_true_at_start_does_not_fire() {
        let res = Model::parse("x' = 1\nx = 0\nat (time >= 0): x = 100")
            .unwrap()
            .run(5.0, &SolverSettings::default());
        assert!(res.events.is_empty());
    }

    /// Antimony's default `fromTrigger = true`: assignments use the values from the
    /// moment the event fires, even when several events fire together.
    #[test]
    fn assignments_use_values_from_the_moment_of_firing() {
        let text = "x' = 0\ny' = 0\nz' = 0\nx = 1; y = 2; z = 0\n\
                    swap: at (time > 1): x = y, y = x\n\
                    copy: at (time > 1): z = x";
        let res = Model::parse(text).unwrap().run(2.0, &SolverSettings::default());
        let last = |i: usize| *res.columns[i].last().unwrap();
        assert!((last(0) - 2.0).abs() < 1e-12 && (last(1) - 1.0).abs() < 1e-12, "swapped");
        assert!((last(2) - 1.0).abs() < 1e-12, "z got the old x, not the swapped one");
    }

    #[test]
    fn events_can_set_off_other_events() {
        let text = "x' = 0\ny' = 0\nx = 0; y = 0\n\
                    first: at (time > 1): x = 1\n\
                    second: at (x > 0.5): y = 5";
        let res = Model::parse(text).unwrap().run(2.0, &SolverSettings::default());
        let names: Vec<&str> = res.events.iter().map(|(_, name)| name.as_str()).collect();
        assert_eq!(names, ["first", "second"]);
        assert_eq!(res.events[0].0, res.events[1].0, "both at the same moment");
        assert!((res.columns[1].last().unwrap() - 5.0).abs() < 1e-12);
    }

    /// Reaction rates are worked out after the run; they must use the parameter
    /// values in force at each time, which events can change.
    #[test]
    fn rates_use_parameter_values_changed_by_events() {
        let text = "J1: $Xo -> S; k*Xo\nJ2: S -> ; S\nXo = 1; k = 1; S = 0\nat (time > 5): k = 0";
        let res = Model::parse(text).unwrap().run(10.0, &SolverSettings::default());
        let j1 = &res.columns[res.names.iter().position(|n| n == "J1").unwrap()];
        assert_eq!(j1[0], 1.0);
        assert_eq!(*j1.last().unwrap(), 0.0);
    }

    #[test]
    fn event_errors() {
        assert!(parse_error("x' = 1\nat (time > 1) after 2: x = 0").contains("delays"));
        assert!(parse_error("x' = 1\nat (time > 1), t0 = false: x = 0").contains("options"));
        assert!(parse_error("x' = 1\nat (time > 1) x = 0").contains("needs ':'"));
        assert!(parse_error("x' = v\nv := 2*x\nat (time > 1): v = 0").contains("calculated"));
        assert!(parse_error("x' = 1\nat (time > 1): q = 0").contains("neither"));
        assert_eq!(parse_error("x' = 1\nat (time >> 1): x = 0"), "Line 2: unexpected '>'");
    }

    #[test]
    fn operator_precedence() {
        assert_eq!(calc("2 + 3*4^2 - -1"), 51.0);
        assert_eq!(calc("2^3^2"), 512.0);
        assert_eq!(calc("-2^2"), -4.0);
        assert_eq!(calc("2**3"), 8.0);
        assert_eq!(calc("(1 + 2) * 3 / 4"), 2.25);
        assert_eq!(calc("1.5e-3 * 2e3"), 3.0);
        assert_eq!(calc("max(1, min(5, 2)) + abs(-1) + pow(2, 3)"), 11.0);
    }

    #[test]
    fn reports_errors_with_line_numbers() {
        assert_eq!(parse_error("x' = -k*x\nk = 1 +"), "Line 2: the expression ends too early");
        assert_eq!(parse_error("x' = -k*x"), "Line 1: unknown name 'k'");
        assert_eq!(parse_error("x' = sin(x"), "Line 1: expected ')' at the end");
        assert_eq!(
            parse_error("x' = -a\na := 2*b\nb := a"),
            "Line 2: 'a' is part of a circular definition"
        );
        assert!(parse_error("x' = -v\nv = 2*x").contains("Use ':='"));
        assert!(parse_error("k = 1").contains("No reactions or rate rules"));
    }

    #[test]
    fn rules_can_be_written_in_any_order() {
        let model = Model::parse("x' = -a\na := 2*b\nb := x\nx = 1").unwrap();
        let sol = model.simulate(1.0, &SolverSettings::rk4(1000));
        // x' = -2x, so x(1) = e^-2.
        let x1 = sol.y.last().unwrap()[0];
        assert!((x1 - (-2.0f64).exp()).abs() < 1e-9, "x(1) = {x1}");
    }

    /// Lotka–Volterra conserves V = δ·x − γ·ln(x) + β·y − α·ln(y),
    /// so an accurate solver should keep V (almost) constant.
    #[test]
    fn lotka_volterra_conserves_invariant() {
        let (a, b, d, g) = (1.1, 0.4, 0.1, 0.4);
        let v = |y: &[f64]| d * y[0] - g * y[0].ln() + b * y[1] - a * y[1].ln();

        let (_, text) = EXAMPLES.iter().find(|(name, _)| *name == "Lotka–Volterra").unwrap();
        let sol = Model::parse(text).unwrap().simulate(50.0, &SolverSettings::rk4(5000));
        let v0 = v(&sol.y[0]);
        let max_drift = sol.y.iter().map(|y| (v(y) - v0).abs()).fold(0.0, f64::max);
        assert!(max_drift < 1e-6, "invariant drifted by {max_drift}");
    }

    /// The pathway's steady state can be worked out by hand:
    /// v1 = 5, so 8·S1/(2 + S1) = 5 gives S1 = 10/3, and 6·S2/(1.5 + S2) = 5 gives S2 = 7.5.
    #[test]
    fn pathway_reaches_steady_state() {
        let (_, text) = EXAMPLES.iter().find(|(name, _)| *name == "Linear pathway").unwrap();
        let model = Model::parse(text).unwrap();
        let names: Vec<&str> = model.species.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, ["S1", "S2"]); // Xo is a boundary species, not simulated

        let end = model.simulate(200.0, &SolverSettings::rk4(5000)).y.last().unwrap().clone();
        assert!((end[0] - 10.0 / 3.0).abs() < 1e-6, "S1 = {}", end[0]);
        assert!((end[1] - 7.5).abs() < 1e-6, "S2 = {}", end[1]);
    }

    /// At the pathway's steady state every reaction carries the same flux, v1 = k1·Xo = 5.
    #[test]
    fn run_reports_reaction_rates() {
        let (_, text) = EXAMPLES.iter().find(|(name, _)| *name == "Linear pathway").unwrap();
        let res = Model::parse(text).unwrap().run(200.0, &SolverSettings::rk4(5000));
        assert_eq!(res.names, ["S1", "S2", "J1", "J2", "J3"]);
        assert_eq!(res.species_count, 2);
        for (name, column) in res.names.iter().zip(&res.columns).skip(2) {
            let flux = *column.last().unwrap();
            assert!((flux - 5.0).abs() < 1e-6, "{name} = {flux}");
        }
    }

    #[test]
    fn every_example_parses_and_simulates() {
        for (name, text) in EXAMPLES {
            let model = Model::parse(text).unwrap_or_else(|e| panic!("{name}: {e}"));
            let sol = model.simulate(50.0, &SolverSettings::default());
            assert!(sol.stopped_early.is_none(), "{name}: {:?}", sol.stopped_early);
            assert_eq!(*sol.t.last().unwrap(), 50.0, "{name} didn't reach the end time");
        }
    }

    /// Reference values for Robertson's problem at t = 40 (Hairer & Wanner,
    /// Solving ODEs II): A = 0.7158270687, B = 9.185534764e-6, C = 0.2841637457.
    #[test]
    fn stiff_solvers_handle_robertson() {
        let (_, text) = EXAMPLES.iter().find(|(name, _)| name.starts_with("Robertson")).unwrap();
        let model = Model::parse(text).unwrap();

        for method in [Method::Bdf, Method::Esdirk34] {
            let settings = SolverSettings { method, rtol: 1e-8, atol: 1e-12, ..Default::default() };
            let sol = model.simulate(40.0, &settings);
            assert!(sol.stopped_early.is_none(), "{method:?}: {:?}", sol.stopped_early);
            let end = sol.y.last().unwrap();
            assert!((end[0] - 0.7158270687).abs() < 1e-5, "{method:?}: A = {}", end[0]);
            assert!((end[1] - 9.185534764e-6).abs() < 1e-9, "{method:?}: B = {}", end[1]);
            assert!((end[2] - 0.2841637457).abs() < 1e-5, "{method:?}: C = {}", end[2]);
            assert!((end.iter().sum::<f64>() - 1.0).abs() < 1e-6, "{method:?}: mass not conserved");
        }

        // The same model defeats fixed-step RK4 at a step size that suits the slow reactions.
        let sol = model.simulate(40.0, &SolverSettings::rk4(5000));
        assert!(sol.stopped_early.is_some(), "RK4 should have blown up");
    }

    /// All four solvers should find the pathway's hand-calculated steady state.
    #[test]
    fn all_solvers_agree_on_pathway_steady_state() {
        let (_, text) = EXAMPLES.iter().find(|(name, _)| *name == "Linear pathway").unwrap();
        let model = Model::parse(text).unwrap();
        for method in Method::ALL {
            let settings = SolverSettings { method, ..Default::default() };
            let end = model.simulate(200.0, &settings).y.last().unwrap().clone();
            assert!((end[0] - 10.0 / 3.0).abs() < 1e-4, "{method:?}: S1 = {}", end[0]);
            assert!((end[1] - 7.5).abs() < 1e-4, "{method:?}: S2 = {}", end[1]);
        }
    }

    #[test]
    fn keeps_slider_values_when_text_changes() {
        let mut old = Model::parse("x' = -k*x\nk = 1\nx = 5").unwrap();
        old.params_mut().next().unwrap().value = 3.0;
        let mut new = Model::parse("x' = -k*x + 0\nk = 1\nx = 6").unwrap();
        new.keep_values_from(&old);
        assert_eq!(new.params_mut().next().unwrap().value, 3.0); // unchanged text: kept
        assert_eq!(new.species[0].value, 6.0); // changed text: new value
    }
}
