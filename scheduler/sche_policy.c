#include "sche_policy.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

void sche_policy_choose(double cpu_energy_j, int cpu_ok,
                        double gpu_energy_j, int gpu_ok,
                        double gpu_power_w, int gp_ok,
                        double budget_w,
                        SchePolicyDecision *out) {

    if (!cpu_ok || !gpu_ok) {
        out->dev = SCHE_DEV_CPU;
        snprintf(out->reason,
                 sizeof(out->reason),
                 "Energy olculmemis ya da NA; güvenli: CPU.");

        return;
    }

    if (budget_w <= 0 || !gp_ok || !isfinite(gpu_power_w) ||
        gpu_power_w <= 0) {
        out->dev = SCHE_DEV_CPU;

        snprintf(
            out->reason,
            sizeof(out->reason),
            "Budget veya GPU güc güvenilir degil; CPU seçildi.");

        return;
    }

    if (gpu_energy_j < cpu_energy_j && gpu_power_w < budget_w) {
        out->dev = SCHE_DEV_GPU;

        snprintf(out->reason,
                 sizeof(out->reason),
                 "GPU enerji (%.6g J) < CPU (%.6g J); GPU güc %.6g W < %.6g W.",
                 gpu_energy_j, cpu_energy_j, gpu_power_w, budget_w);
        return;
    }

    out->dev = SCHE_DEV_CPU;

    if (gpu_energy_j >= cpu_energy_j && gpu_power_w >= budget_w) {
        snprintf(out->reason, sizeof(out->reason),
                 "GPU enerji CPU dan düsük degil VE GPU gücü budget üst/esit.");

    } else if (gpu_energy_j >= cpu_energy_j) {

        snprintf(out->reason,
                 sizeof(out->reason),

                 "GPU enerji daha yüksek/esit.");

    } else {

        snprintf(out->reason,

                 sizeof(out->reason),

                 "GPU güc %.6g W >= budget %.6g W.",

                 gpu_power_w, budget_w);
    }

}
