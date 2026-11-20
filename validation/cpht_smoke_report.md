# Pure CPHT comparison

Reports: 120. Missing: 0. Smoke only: True.

Task success is omitted when every episode is budget-censored.

| Condition/method | Completion | Speed RMSE | Mean decision seconds | Flags |
| --- | --- | --- | --- | --- |
| sensed_belief/FtoF/h0.02/fixed/friction_change/cem | n/a | 2.989 | 0.02487 | evaluation_budget_censored, no_informative_friction_evidence |
| sensed_belief/FtoF/h0.02/fixed/friction_change/selector | n/a | 2.986 | 0.01675 | evaluation_budget_censored |
| sensed_belief/FtoF/h0.02/fixed/friction_change/selector_value | n/a | 2.992 | 0.0122 | evaluation_budget_censored |
| sensed_belief/FtoF/h0.02/fixed/friction_change/shooting | n/a | 2.963 | 0.0534 | evaluation_budget_censored |
| sensed_belief/FtoF/h0.02/fixed/nominal/cem | n/a | 3.026 | 0.01564 | evaluation_budget_censored |
| sensed_belief/FtoF/h0.02/fixed/nominal/selector | n/a | 2.996 | 0.02864 | evaluation_budget_censored, no_informative_friction_evidence |
| sensed_belief/FtoF/h0.02/fixed/nominal/selector_value | n/a | 2.996 | 0.04627 | evaluation_budget_censored |
| sensed_belief/FtoF/h0.02/fixed/nominal/shooting | n/a | 2.961 | 0.04325 | evaluation_budget_censored |
| sensed_belief/FtoF/h0.02/fixed/sensor_shift/cem | n/a | 3.026 | 0.06805 | evaluation_budget_censored |
| sensed_belief/FtoF/h0.02/fixed/sensor_shift/selector | n/a | 2.996 | 0.01078 | evaluation_budget_censored, no_informative_friction_evidence |
| sensed_belief/FtoF/h0.02/fixed/sensor_shift/selector_value | n/a | 2.996 | 0.009398 | evaluation_budget_censored |
| sensed_belief/FtoF/h0.02/fixed/sensor_shift/shooting | n/a | 2.959 | 0.04411 | evaluation_budget_censored |
| sensed_belief/FtoF/h0.02/irregular/friction_change/cem | n/a | 2.992 | 0.01544 | evaluation_budget_censored |
| sensed_belief/FtoF/h0.02/irregular/friction_change/selector | n/a | 2.993 | 0.01458 | evaluation_budget_censored |
| sensed_belief/FtoF/h0.02/irregular/friction_change/selector_value | n/a | 2.991 | 0.01005 | evaluation_budget_censored |
| sensed_belief/FtoF/h0.02/irregular/friction_change/shooting | n/a | 2.973 | 0.05196 | evaluation_budget_censored |
| sensed_belief/FtoF/h0.02/irregular/nominal/cem | n/a | 2.992 | 0.02562 | evaluation_budget_censored |
| sensed_belief/FtoF/h0.02/irregular/nominal/selector | n/a | 2.971 | 0.01284 | evaluation_budget_censored |
| sensed_belief/FtoF/h0.02/irregular/nominal/selector_value | n/a | 2.99 | 0.009595 | evaluation_budget_censored |
| sensed_belief/FtoF/h0.02/irregular/nominal/shooting | n/a | 2.972 | 0.05375 | evaluation_budget_censored |
| sensed_belief/FtoF/h0.02/irregular/sensor_shift/cem | n/a | 2.992 | 0.01569 | evaluation_budget_censored |
| sensed_belief/FtoF/h0.02/irregular/sensor_shift/selector | n/a | 2.971 | 0.03247 | evaluation_budget_censored |
| sensed_belief/FtoF/h0.02/irregular/sensor_shift/selector_value | n/a | 2.99 | 0.009236 | evaluation_budget_censored |
| sensed_belief/FtoF/h0.02/irregular/sensor_shift/shooting | n/a | 2.975 | 0.05312 | evaluation_budget_censored |
| sensed_belief/FtoJ/h0.02/fixed/friction_change/cem | n/a | 2.989 | 0.01398 | evaluation_budget_censored, no_informative_friction_evidence |
| sensed_belief/FtoJ/h0.02/fixed/friction_change/selector | n/a | 2.986 | 0.03283 | evaluation_budget_censored |
| sensed_belief/FtoJ/h0.02/fixed/friction_change/selector_value | n/a | 2.992 | 0.02461 | evaluation_budget_censored |
| sensed_belief/FtoJ/h0.02/fixed/friction_change/shooting | n/a | 2.963 | 0.04587 | evaluation_budget_censored |
| sensed_belief/FtoJ/h0.02/fixed/nominal/cem | n/a | 3.026 | 0.01366 | evaluation_budget_censored |
| sensed_belief/FtoJ/h0.02/fixed/nominal/selector | n/a | 2.996 | 0.0328 | evaluation_budget_censored, no_informative_friction_evidence |
| sensed_belief/FtoJ/h0.02/fixed/nominal/selector_value | n/a | 2.996 | 0.01759 | evaluation_budget_censored |
| sensed_belief/FtoJ/h0.02/fixed/nominal/shooting | n/a | 2.961 | 0.04242 | evaluation_budget_censored |
| sensed_belief/FtoJ/h0.02/fixed/sensor_shift/cem | n/a | 3.026 | 0.01413 | evaluation_budget_censored |
| sensed_belief/FtoJ/h0.02/fixed/sensor_shift/selector | n/a | 2.996 | 0.01487 | evaluation_budget_censored, no_informative_friction_evidence |
| sensed_belief/FtoJ/h0.02/fixed/sensor_shift/selector_value | n/a | 2.996 | 0.01269 | evaluation_budget_censored |
| sensed_belief/FtoJ/h0.02/fixed/sensor_shift/shooting | n/a | 2.959 | 0.048 | evaluation_budget_censored |
| sensed_belief/FtoJ/h0.02/irregular/friction_change/cem | n/a | 2.992 | 0.01592 | evaluation_budget_censored |
| sensed_belief/FtoJ/h0.02/irregular/friction_change/selector | n/a | 2.993 | 0.0111 | evaluation_budget_censored |
| sensed_belief/FtoJ/h0.02/irregular/friction_change/selector_value | n/a | 2.991 | 0.01102 | evaluation_budget_censored |
| sensed_belief/FtoJ/h0.02/irregular/friction_change/shooting | n/a | 2.973 | 0.1492 | evaluation_budget_censored |
| sensed_belief/FtoJ/h0.02/irregular/nominal/cem | n/a | 2.992 | 0.01497 | evaluation_budget_censored |
| sensed_belief/FtoJ/h0.02/irregular/nominal/selector | n/a | 2.971 | 0.01483 | evaluation_budget_censored |
| sensed_belief/FtoJ/h0.02/irregular/nominal/selector_value | n/a | 2.99 | 0.01316 | evaluation_budget_censored |
| sensed_belief/FtoJ/h0.02/irregular/nominal/shooting | n/a | 2.972 | 1.161 | evaluation_budget_censored |
| sensed_belief/FtoJ/h0.02/irregular/sensor_shift/cem | n/a | 2.992 | 0.0171 | evaluation_budget_censored |
| sensed_belief/FtoJ/h0.02/irregular/sensor_shift/selector | n/a | 2.971 | 0.01643 | evaluation_budget_censored |
| sensed_belief/FtoJ/h0.02/irregular/sensor_shift/selector_value | n/a | 2.99 | 0.0157 | evaluation_budget_censored |
| sensed_belief/FtoJ/h0.02/irregular/sensor_shift/shooting | n/a | 2.975 | 0.1026 | evaluation_budget_censored |
| sensed_belief/JtoJ/h0.02/fixed/friction_change/cem | n/a | 2.952 | 0.01869 | evaluation_budget_censored |
| sensed_belief/JtoJ/h0.02/fixed/friction_change/selector | n/a | 2.968 | 0.03338 | evaluation_budget_censored |
| sensed_belief/JtoJ/h0.02/fixed/friction_change/selector_value | n/a | 2.985 | 0.01528 | evaluation_budget_censored |
| sensed_belief/JtoJ/h0.02/fixed/friction_change/shooting | n/a | 2.958 | 0.05278 | evaluation_budget_censored |
| sensed_belief/JtoJ/h0.02/fixed/nominal/cem | n/a | 2.945 | 0.02033 | evaluation_budget_censored |
| sensed_belief/JtoJ/h0.02/fixed/nominal/selector | n/a | 2.94 | 0.0148 | evaluation_budget_censored |
| sensed_belief/JtoJ/h0.02/fixed/nominal/selector_value | n/a | 2.974 | 0.01452 | evaluation_budget_censored |
| sensed_belief/JtoJ/h0.02/fixed/nominal/shooting | n/a | 2.957 | 0.08633 | evaluation_budget_censored |
| sensed_belief/JtoJ/h0.02/fixed/sensor_shift/cem | n/a | 2.945 | 0.01802 | evaluation_budget_censored |
| sensed_belief/JtoJ/h0.02/fixed/sensor_shift/selector | n/a | 2.941 | 0.01572 | evaluation_budget_censored |
| sensed_belief/JtoJ/h0.02/fixed/sensor_shift/selector_value | n/a | 2.974 | 0.01465 | evaluation_budget_censored |
| sensed_belief/JtoJ/h0.02/fixed/sensor_shift/shooting | n/a | 2.957 | 0.04774 | evaluation_budget_censored |
| sensed_belief/JtoJ/h0.02/irregular/friction_change/cem | n/a | 2.961 | 0.04295 | evaluation_budget_censored, no_informative_friction_evidence |
| sensed_belief/JtoJ/h0.02/irregular/friction_change/selector | n/a | 2.988 | 0.01473 | evaluation_budget_censored |
| sensed_belief/JtoJ/h0.02/irregular/friction_change/selector_value | n/a | 2.988 | 0.01337 | evaluation_budget_censored |
| sensed_belief/JtoJ/h0.02/irregular/friction_change/shooting | n/a | 2.963 | 0.09237 | evaluation_budget_censored |
| sensed_belief/JtoJ/h0.02/irregular/nominal/cem | n/a | 2.943 | 0.09026 | evaluation_budget_censored, no_informative_friction_evidence |
| sensed_belief/JtoJ/h0.02/irregular/nominal/selector | n/a | 2.981 | 0.01378 | evaluation_budget_censored, no_informative_friction_evidence |
| sensed_belief/JtoJ/h0.02/irregular/nominal/selector_value | n/a | 2.982 | 0.0131 | evaluation_budget_censored |
| sensed_belief/JtoJ/h0.02/irregular/nominal/shooting | n/a | 2.954 | 0.1213 | evaluation_budget_censored, no_informative_friction_evidence |
| sensed_belief/JtoJ/h0.02/irregular/sensor_shift/cem | n/a | 2.943 | 0.05159 | evaluation_budget_censored, no_informative_friction_evidence |
| sensed_belief/JtoJ/h0.02/irregular/sensor_shift/selector | n/a | 2.981 | 0.01733 | evaluation_budget_censored, no_informative_friction_evidence |
| sensed_belief/JtoJ/h0.02/irregular/sensor_shift/selector_value | n/a | 2.983 | 0.01537 | evaluation_budget_censored |
| sensed_belief/JtoJ/h0.02/irregular/sensor_shift/shooting | n/a | 2.953 | 0.0749 | evaluation_budget_censored, no_informative_friction_evidence |
| state_belief/FtoF/h0.02/fixed/friction_change/cem | n/a | 2.989 | 0.02383 | evaluation_budget_censored, no_informative_friction_evidence |
| state_belief/FtoF/h0.02/fixed/friction_change/selector | n/a | 2.991 | 0.01256 | evaluation_budget_censored |
| state_belief/FtoF/h0.02/fixed/friction_change/selector_value | n/a | 2.993 | 0.01366 | evaluation_budget_censored, no_informative_friction_evidence |
| state_belief/FtoF/h0.02/fixed/friction_change/shooting | n/a | 2.972 | 0.05619 | evaluation_budget_censored |
| state_belief/FtoF/h0.02/fixed/nominal/cem | n/a | 3.026 | 0.01596 | evaluation_budget_censored |
| state_belief/FtoF/h0.02/fixed/nominal/selector | n/a | 2.993 | 0.01152 | evaluation_budget_censored, no_informative_friction_evidence |
| state_belief/FtoF/h0.02/fixed/nominal/selector_value | n/a | 2.995 | 0.0137 | evaluation_budget_censored |
| state_belief/FtoF/h0.02/fixed/nominal/shooting | n/a | 2.966 | 0.2487 | evaluation_budget_censored |
| state_belief/FtoF/h0.02/irregular/friction_change/cem | n/a | 2.992 | 0.02343 | evaluation_budget_censored |
| state_belief/FtoF/h0.02/irregular/friction_change/selector | n/a | 2.992 | 0.01064 | evaluation_budget_censored |
| state_belief/FtoF/h0.02/irregular/friction_change/selector_value | n/a | 2.99 | 0.01196 | evaluation_budget_censored |
| state_belief/FtoF/h0.02/irregular/friction_change/shooting | n/a | 2.974 | 0.06329 | evaluation_budget_censored, no_informative_friction_evidence |
| state_belief/FtoF/h0.02/irregular/nominal/cem | n/a | 2.992 | 0.01784 | evaluation_budget_censored |
| state_belief/FtoF/h0.02/irregular/nominal/selector | n/a | 2.97 | 0.01051 | evaluation_budget_censored |
| state_belief/FtoF/h0.02/irregular/nominal/selector_value | n/a | 2.989 | 0.01133 | evaluation_budget_censored |
| state_belief/FtoF/h0.02/irregular/nominal/shooting | n/a | 2.967 | 0.0773 | evaluation_budget_censored, no_informative_friction_evidence |
| state_belief/FtoJ/h0.02/fixed/friction_change/cem | n/a | 2.989 | 0.01524 | evaluation_budget_censored, no_informative_friction_evidence |
| state_belief/FtoJ/h0.02/fixed/friction_change/selector | n/a | 2.991 | 0.01112 | evaluation_budget_censored |
| state_belief/FtoJ/h0.02/fixed/friction_change/selector_value | n/a | 2.993 | 0.01001 | evaluation_budget_censored, no_informative_friction_evidence |
| state_belief/FtoJ/h0.02/fixed/friction_change/shooting | n/a | 2.972 | 0.05755 | evaluation_budget_censored |
| state_belief/FtoJ/h0.02/fixed/nominal/cem | n/a | 3.026 | 0.01785 | evaluation_budget_censored |
| state_belief/FtoJ/h0.02/fixed/nominal/selector | n/a | 2.993 | 0.01085 | evaluation_budget_censored, no_informative_friction_evidence |
| state_belief/FtoJ/h0.02/fixed/nominal/selector_value | n/a | 2.995 | 0.0102 | evaluation_budget_censored |
| state_belief/FtoJ/h0.02/fixed/nominal/shooting | n/a | 2.966 | 0.05045 | evaluation_budget_censored |
| state_belief/FtoJ/h0.02/irregular/friction_change/cem | n/a | 2.992 | 0.04734 | evaluation_budget_censored |
| state_belief/FtoJ/h0.02/irregular/friction_change/selector | n/a | 2.992 | 0.01398 | evaluation_budget_censored |
| state_belief/FtoJ/h0.02/irregular/friction_change/selector_value | n/a | 2.99 | 0.01496 | evaluation_budget_censored |
| state_belief/FtoJ/h0.02/irregular/friction_change/shooting | n/a | 2.974 | 0.06113 | evaluation_budget_censored, no_informative_friction_evidence |
| state_belief/FtoJ/h0.02/irregular/nominal/cem | n/a | 2.992 | 0.01595 | evaluation_budget_censored |
| state_belief/FtoJ/h0.02/irregular/nominal/selector | n/a | 2.97 | 0.01032 | evaluation_budget_censored |
| state_belief/FtoJ/h0.02/irregular/nominal/selector_value | n/a | 2.989 | 0.01343 | evaluation_budget_censored |
| state_belief/FtoJ/h0.02/irregular/nominal/shooting | n/a | 2.967 | 0.05792 | evaluation_budget_censored, no_informative_friction_evidence |
| state_belief/JtoJ/h0.02/fixed/friction_change/cem | n/a | 2.947 | 0.0226 | evaluation_budget_censored |
| state_belief/JtoJ/h0.02/fixed/friction_change/selector | n/a | 2.988 | 0.04412 | evaluation_budget_censored |
| state_belief/JtoJ/h0.02/fixed/friction_change/selector_value | n/a | 2.968 | 0.01412 | evaluation_budget_censored |
| state_belief/JtoJ/h0.02/fixed/friction_change/shooting | n/a | 2.958 | 0.07388 | evaluation_budget_censored |
| state_belief/JtoJ/h0.02/fixed/nominal/cem | n/a | 2.945 | 0.02041 | evaluation_budget_censored |
| state_belief/JtoJ/h0.02/fixed/nominal/selector | n/a | 2.977 | 0.02312 | evaluation_budget_censored, no_informative_friction_evidence |
| state_belief/JtoJ/h0.02/fixed/nominal/selector_value | n/a | 2.929 | 0.01045 | evaluation_budget_censored, large_one_step_model_error |
| state_belief/JtoJ/h0.02/fixed/nominal/shooting | n/a | 2.957 | 0.06977 | evaluation_budget_censored |
| state_belief/JtoJ/h0.02/irregular/friction_change/cem | n/a | 2.96 | 0.02226 | evaluation_budget_censored, no_informative_friction_evidence |
| state_belief/JtoJ/h0.02/irregular/friction_change/selector | n/a | 2.988 | 0.0159 | evaluation_budget_censored |
| state_belief/JtoJ/h0.02/irregular/friction_change/selector_value | n/a | 2.99 | 0.01162 | evaluation_budget_censored |
| state_belief/JtoJ/h0.02/irregular/friction_change/shooting | n/a | 2.964 | 0.07035 | evaluation_budget_censored, no_informative_friction_evidence |
| state_belief/JtoJ/h0.02/irregular/nominal/cem | n/a | 2.943 | 0.0189 | evaluation_budget_censored, no_informative_friction_evidence |
| state_belief/JtoJ/h0.02/irregular/nominal/selector | n/a | 2.983 | 0.01744 | evaluation_budget_censored |
| state_belief/JtoJ/h0.02/irregular/nominal/selector_value | n/a | 2.986 | 0.009417 | evaluation_budget_censored |
| state_belief/JtoJ/h0.02/irregular/nominal/shooting | n/a | 2.954 | 0.07269 | evaluation_budget_censored, no_informative_friction_evidence |

No completed-course or generalization claim follows from a smoke/censored run.
