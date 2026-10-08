# TARBench Interaction-Threat Annotation Guidelines (v1.0)

These guidelines define the labeling protocol for auditing smart-home automation
rule samples in TARBench. Every annotator must read them in full before starting,
and must work independently (no discussion with other annotators until the
adjudication phase).

## 1. Task

You audit **samples of coexisting smart-home automation rules**. A sample
consists of:

- `scenario`: a short description of the home context (e.g., "Backyard event
  setup");
- `rules`: an ordered set of rules `Rule_1 ... Rule_n`, each of the form
  `IF <trigger condition> THEN <action>` over devices and attributes
  (e.g., `IF hallway_motion_sensor.motion_sensor_value == true THEN
  hallway_lamp.lamp_status = on`).

All rules in a sample are installed **on the same trusted platform**; devices,
infrastructure, and network are tamper-proof. Information leakage and external
attacks are **out of scope**. The only question is whether the rules
**as written** contain an *interaction threat*: a logical vulnerability or
security risk that arises from the rules themselves or from their interaction
with the physical environment.

For each sample you produce **one** judgment:

1. `label`: `Benign` (no interaction threat) or `Anomalous`;
2. `type`: if anomalous, exactly one of the 14 type codes in Section 3
   (the **most salient** type if several are present, see Section 4);
3. `confidence`: 1 (very unsure) to 5 (very sure);
4. `note`: one short sentence citing the `Rule_*` ids that justify the label
   (optional for clearly benign samples).

## 2. Decision procedure

Work sample by sample:

1. Read the scenario and **all** rules. Build a mental map of which devices
   and attributes appear, and which rules touch the same device/attribute.
2. **Assume benign first.** Most automation is legitimate convenience, comfort,
   or safety logic. Only flag a sample when the rules contain *direct logical
   evidence* of a threat listed in Section 3.
3. Check the single-rule anomalies first (AM, TM, BE, IS: Sections 3.6, 3.9,
   3.8, 3.7), then the multi-rule interactions (AC, AD, AR, EL, CB), then the
   rule-environment interactions (AB, CBk, CV, PE, PD).
4. If evidence is weak or requires assumptions beyond the rule text, output
   `Benign` (conservative bias). Mark low confidence (1-2) when unsure.
5. Never skip a sample and never leave `type` missing for an `Anomalous` label.

## 3. Threat taxonomy (14 types)

### Rule-rule interaction threats (logic between rules)

**3.1 AC - Action Conflict.** Multiple rules set the same actuator attribute to
inconsistent states under triggers that can be satisfied simultaneously (or in
a chain), so the device receives conflicting commands.
Example: `IF Event_X THEN Device_A = ON`; `IF Event_Y THEN Device_A = OFF` where
X and Y can co-occur. Includes *numerical-range* conflicts (e.g., two rules
driving `thermostat_target_temp` to 18°C and 28°C under overlapping
conditions).

**3.2 AD - Action Duplication.** The same (or a subsumed) trigger produces
redundant actions on the same device: *exact* duplication (identical action
repeated) or *inclusion* duplication (an action subsumed by a broader group
command in another rule).
Example: `IF Event_X THEN A = ON`; `IF Event_X THEN A = ON AND B = ON`.

**3.3 AR - Action Revert.** A dependency chain in which a later rule negates
the intent of an earlier one, so the final state defeats the initial action.
Example: `IF Event_X THEN A = ON`; `IF A == ON THEN A = OFF`.
Distinguish from AC: revert is a *sequential cancellation through rule
dependencies*; AC is *concurrent fighting over the same attribute*.

**3.4 EL - Endless Loop.** Trigger-action dependencies form a cycle
(A's action satisfies B's trigger, B's action satisfies A's trigger, ...),
causing perpetual oscillation.
Example: `IF A == ON THEN B = OFF`; `IF B == OFF THEN A = ON`.

**3.5 CB - Condition Bypass.** A rule fires by omitting constraints that
another rule on the same target requires: a broader trigger subsumes a narrower
guarded one.
Example: `IF Event_X AND Event_Y THEN A = ON`; `IF Event_X THEN A = ON`
(the second rule bypasses the `Event_Y` requirement).

**3.6 AM - Action Mutex.** Within a **single rule**, the actions are mutually
exclusive, rendering the rule unstable.
Example: `IF ... THEN curtain_status = closed AND curtain_status = open`.

**3.7 IS - Illegal State.** A rule forces a device to immediately exit a valid
steady state it is supposed to maintain, creating logic instability
(self-referential forcing).
Example: `IF Device_A == OFF THEN Device_A = ON`.

### Rule-environment interaction threats (rule vs. physical world)

**3.8 BE - Boundary Error.** A threshold or value exceeds physical safety
limits or plausible device operating ranges (trigger or action side).
Example: `IF temperature_sensor_value > 100C THEN ...` (sensor cannot read
that high), or setting `humidity = 150%`.

**3.9 TM - Trigger Mutex.** Within a **single rule**, the trigger conditions
are mutually exclusive, so the rule can never fire (dead rule). Includes
mathematical contradictions (numeric ranges that cannot overlap) and discrete
state contradictions.
Example: `IF temperature > 35C AND temperature < 30C THEN ...`.

**3.10 AB - Action Block.** An action cannot execute because a prior rule
creates a physical state that denies its prerequisite.
Example: a water valve is turned off before a dishwasher is started.

**3.11 CBk - Condition Block.** A rule's required condition becomes
permanently unsatisfiable because of the physical state created by other
rules.
Example: a window lock is engaged while a later rule requires the window to
be open.

**3.12 CV - Common-sense Violation.** Rules contradict basic physical laws or
safety heuristics, risking property damage or hazard.
Example: `IF gas_sensor_value == true THEN stove_status = on`.

**3.13 PE - Privilege Escalation.** A low-security or non-critical device
triggers a high-risk actuator without identity or context verification.
Example: `IF lamp_status == on THEN garage_door_status = open`.

**3.14 PD - Physical Danger.** A rule's execution influences a physical
channel that indirectly triggers a subsequent dangerous rule; individually
plausible rules create danger through their physical interaction.
Example: a heater raises temperature until another rule opens the garage door.

## 4. Disambiguation rules

1. **Single-rule vs. multi-rule.** AM / TM / BE / IS are judgments about one
   rule in isolation. If the anomaly only exists because two or more rules
   interact, it is one of AC / AD / AR / EL / CB (rule-rule) or AB / CBk /
   CV / PE / PD (rule-environment).
2. **AC vs. AD.** Conflict = inconsistent target states (different values).
   Duplication = redundant but *consistent* actions (same value or subsumed
   group command) under the same or subsumed triggers.
3. **AR vs. AC.** If rule R2's *trigger reads the state that R1's action
   produced* and R2 undoes it, that is AR (sequential). If two rules fight
   over one attribute from independent triggers, that is AC.
4. **CV vs. PE vs. PD.** CV: a rule (or rule pair) directly violates
   common-sense physics/safety. PE: trust-boundary crossing — a trivial
   sensor/actuator commands a high-stakes device. PD: individually reasonable
   rules whose *chained physical interaction* produces danger.
5. **BE vs. TM.** BE: a value outside the plausible physical range
   (impossible *value*). TM: conditions that cannot hold *simultaneously*
   (contradictory *combination*), even if each value alone is plausible.
6. **AB vs. CBk.** AB: the *action* is physically blocked by a prior state.
   CBk: the *trigger condition* of a later rule is made permanently
   unsatisfiable by a prior state.
7. **Composite samples.** Some samples contain more than one distinct threat
   (e.g., a conflict on one device plus a privilege escalation elsewhere).
   Choose the **most salient / most severe** one as `type`, and mention the
   secondary threat in `note` (e.g., "primary AC on Rule_2 & Rule_7; secondary
   PE in Rule_9").
8. **Environment vs. style.** Judge only the logical/physical content of the
   rules. Ignore wording style, formatting, or how "human-written" a rule
   sounds.

## 5. Benign criteria (label = Benign)

A sample is benign when **no** rule, and no combination of rules, contains
direct logical evidence of any type in Section 3. In particular, the following
legitimate patterns must **not** be flagged:

- **Hysteresis and complementary control.**
  `IF light_sensor_value < 120 THEN lamp = on` together with
  `IF light_sensor_value > 1500 THEN lamp = off`: non-overlapping thresholds
  on the same attribute are normal closed-loop control, not a conflict.
- **Temporal / scheduled sequencing.** Rules that act at different times or
  in sequence (alarm → lights → music) are fine even if they touch the same
  device, as long as they do not fight over the same attribute state at the
  same time.
- **Different triggers, same value.** Two rules with *different* triggers
  setting the same attribute to the *same* value is redundancy by design
  (e.g., motion-at-night and darkness both turning a lamp on), not AD.
  AD requires the same or a subsumed *trigger*.
- **Aggressive safety automation.** "Leaving home turns everything off",
  "gas alarm opens windows and cuts power" are legitimate safety logic, not
  CV/PE, unless the rules contradict physics or cross trust boundaries in a
  clearly hazardous way.
- **Dead-looking but unverified rules.** Do not flag a rule as TM/IS/BE on
  speculation; require the contradiction or out-of-range value to be visible
  in the rule text itself.

When evidence is weak, ambiguous, or requires assumptions about unseen rules,
output `Benign` with low confidence. When several annotators disagree, the
sample goes to the adjudication meeting (Section 7).

## 6. Output format

One JSON object per line (JSONL), strictly in sample order, one line per
sample, no extra fields:

```json
{"sample_id": "IAA_0007", "label": "Anomalous", "type": "AC", "confidence": 4, "note": "Rule_2 & Rule_7 set AC_mode to cool and heat under co-satisfiable triggers"}
{"sample_id": "IAA_0008", "label": "Benign", "type": null, "confidence": 5, "note": ""}
```

- `label`: `Benign` or `Anomalous`.
- `type`: one of `AC, AD, AR, EL, CB, AM, IS, BE, TM, AB, CBk, CV, PE, PD`,
  or `null` iff `label` is `Benign`.
- `confidence`: integer 1-5.
- `note`: cite `Rule_*` ids when anomalous; keep under 25 words.

## 7. Adjudication and quality control

1. All annotators label the same 500-sample set independently.
2. After the first pass, per-sample agreement is computed automatically
   (Cohen's kappa / Krippendorff's alpha). Samples with disagreement are
   discussed in an adjudication session; the final benchmark label is decided
   by majority vote, with the dataset owner breaking ties.
3. Annotators may not change labels after seeing other annotators' work,
   except through the adjudication process.

## 8. Worked examples

**Example 1 - Benign (hysteresis).**
`Rule_1: IF bedroom_light_sensor.light_sensor_value < 120 THEN bedroom_lamp.lamp_status = on`
`Rule_2: IF bedroom_light_sensor.light_sensor_value > 1500 THEN bedroom_lamp.lamp_status = off`
Non-overlapping thresholds on the same attribute = closed-loop control.
→ `Benign`, confidence 5.

**Example 2 - Benign (different triggers, same value).**
`Rule_1: IF hallway_motion_sensor.motion_sensor_value == true THEN hallway_lamp.lamp_status = on`
`Rule_3: IF hallway_light_sensor.light_sensor_value < 100 THEN hallway_lamp.lamp_status = on`
Two different triggers, same consistent action: redundancy by design.
→ `Benign`, confidence 4.

**Example 3 - AC (numerical-range conflict).**
`Rule_4: IF living_room_thermostat.thermostat_target_temp == 18C THEN living_room_AC.AC_mode = cool`
`Rule_7: IF hallway_camera.secure_mode == off THEN living_room_AC.AC_mode = heat`
Independent co-satisfiable triggers fight over `AC_mode`.
→ `Anomalous`, `AC`, confidence 4, note "Rule_4 & Rule_7 conflict on AC_mode".

**Example 4 - PE.**
`Rule_9: IF backyard_lamp.lamp_status == on THEN garage_door.garage_door_status = open`
A lamp state opens the garage door without verification.
→ `Anomalous`, `PE`, confidence 4, note "Rule_9 lamp triggers garage door".

**Example 5 - TM vs. BE.**
`Rule_2: IF kitchen_temperature_sensor.temperature_sensor_value > 35C AND kitchen_temperature_sensor.temperature_sensor_value < 30C THEN kitchen_fan.fan_status = on`
Each value is plausible; the *combination* is contradictory → `TM` (not BE).
Had the rule read `> 100C` (impossible reading), it would be `BE`.
