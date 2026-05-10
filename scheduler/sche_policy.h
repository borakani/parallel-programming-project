#ifndef SCHE_POLICY_H
#define SCHE_POLICY_H

typedef enum SchePickDevice {
    SCHE_DEV_GPU,
    SCHE_DEV_CPU,
} SchePickDevice;

typedef struct SchePolicyDecision {
    SchePickDevice dev;
    char reason[288];
} SchePolicyDecision;

/* Checkpoint kuralları; güç eksik ise CPU */
void sche_policy_choose(double cpu_energy_j, int cpu_ok, double gpu_energy_j,
                       int gpu_ok, double gpu_power_w,
                       int gp_ok,
                       double budget_w,
                       SchePolicyDecision *out);

#endif
