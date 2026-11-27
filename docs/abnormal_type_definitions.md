1. Action Conflict: Multiple rules trigger inconsistent states on the same actuator simultaneously.
- Example: IF Event_X THEN Device_A = ON; IF Event_X THEN Device_A = OFF.

2. Action Duplication: Redundant actions triggered on the same device, or an action subsumed by a broader group command.
- Example: IF Event_X THEN Device_A = ON; IF Event_X THEN Device_A = ON AND Device_B = ON.

3. Action Revert: A rule chain where the final action negates the initial rule's intent.
- Example: IF Event_X THEN Device_A = ON; IF Device_A == ON THEN Device_A = OFF.

4. Loop: Causal dependencies between rules form a closed loop, leading to infinite state oscillation.
- Example: IF Device_A == ON THEN Device_B = OFF; IF Device_B == OFF THEN Device_A = ON.

5. Condition Bypass: A rule triggers an action by omitting specific constraints or logic required by another rule.
- Example: IF Event_X AND Event_Y THEN Device_A = ON; IF Event_X THEN Device_A = ON.

6. Action Block: An action cannot execute because a prior rule creates a physical state that denies necessary prerequisites.
- Example: IF ... THEN Water Valve = OFF; IF ... THEN dishwasher = ON. (The action is blocked by no water flow)

7. Condition Block: A rule’s required condition becomes permanently unsatisfiable due to the physical state.
- Example: IF ...THEN window_lock_status = LOCKED; IF Window == OPEN THEN ...(The "window == open" condition can never be met while the lock is engaged)

8. Safety Boundary Error: Thresholds that exceed physical safety limits or device operating ranges.
- Example: IF temperature_sensor_value > 100°C THEN...

9. Trigger Mutex: A single rule contains mutually exclusive logic in its triggers, rendering it dead.
- Example: IF temperature_sensor_value > 35°C AND temperature_sensor_value < 30°C THEN...

10. Action Mutex: A single rule contains mutually exclusive logic in its actions, rendering it unstable.
- Example: IF ... THEN curtain_status = closed AND curtain_status = open.

11. Illegal State Rule: A rule that creates logic instability by forcing a device to immediately exit a valid steady state, making that state impossible to maintain.
- Example: IF Device_A == OFF THEN Device_A = ON.

12. Common Sense Violation: Rules that contradict basic physical laws or safety heuristics, leading to property damage.
- Example: IF gas_sensor_value == true THEN stove_status = on.

13. Privilege Escalation: Low-security or non-critical devices triggering high-risk actuators without identity or context verification.
- Example: IF lamp_status == on THEN garage_door_status = open.

14. Implicit Physical Attack (Chaining): A rule execution influences a physical channel (temp, humidity, etc.), which indirectly triggers a subsequent dangerous rule.
- Example: Rule 1: IF temperature_sensor_value < 10°C THEN heater_status = on;Rule 2:IF temperature_sensor_value > 35°C THEN garage_door_status = open. (Rule 1 induces the environment to trigger Rule 2)
