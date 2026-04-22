# Reviewer Supplementary Experiments

## Scope
These experiments were run on the current repository and current local dataset (2,029 observations; 123 firms for model training/calibration; 32 firms for portfolio validation).

## Multicollinearity Check
Pairwise correlations among the five core indicators:

```text
                             currentRatio  netProfitMargin  assetTurnover  operatingCashFlowSalesRatio  debtEquityRatio
currentRatio                        1.000           -0.012         -0.072                        0.054           -0.043
netProfitMargin                    -0.012            1.000         -0.048                        0.174            0.003
assetTurnover                      -0.072           -0.048          1.000                       -0.228            0.003
operatingCashFlowSalesRatio         0.054            0.174         -0.228                        1.000           -0.031
debtEquityRatio                    -0.043            0.003          0.003                       -0.031            1.000
```

Variance inflation factors:

```text
                   variable     r2   vif
               currentRatio 0.0089 1.009
            netProfitMargin 0.0309 1.032
              assetTurnover 0.0559 1.059
operatingCashFlowSalesRatio 0.0813 1.088
            debtEquityRatio 0.0028 1.003
```

## Optimization Robustness and Benchmarking
Matched evaluation budget for all algorithms: `800` objective evaluations per run.

Summary statistics (mean ± sd across seeds):

```text
algorithm          metric         mean          std
   MOEA/D       runtime_s 1.654482e-01 3.887283e-03
  NSGA-II       runtime_s 6.873033e-02 5.468973e-03
    SA-NA       runtime_s 4.116337e-02 3.359092e-03
  SA-only       runtime_s 3.975899e-02 9.151219e-04
    SPEA2       runtime_s 7.535551e-02 1.001243e-03
   MOEA/D objective_calls 8.000000e+02 0.000000e+00
  NSGA-II objective_calls 8.000000e+02 0.000000e+00
    SA-NA objective_calls 8.000000e+02 0.000000e+00
  SA-only objective_calls 8.000000e+02 0.000000e+00
    SPEA2 objective_calls 8.000000e+02 0.000000e+00
   MOEA/D     pareto_size 5.050000e+01 2.413504e+01
  NSGA-II     pareto_size 3.800000e+00 1.686548e+00
    SA-NA     pareto_size 2.100000e+00 1.523884e+00
  SA-only     pareto_size 1.900000e+00 5.676462e-01
    SPEA2     pareto_size 4.600000e+00 1.505545e+00
   MOEA/D      best_raroc 7.064174e-01 3.726457e-01
  NSGA-II      best_raroc 1.249383e+00 2.424209e-01
    SA-NA      best_raroc 8.448666e-01 6.204975e-02
  SA-only      best_raroc 7.592522e-01 2.323422e-01
    SPEA2      best_raroc 1.263304e+00 1.934853e-01
   MOEA/D       best_cvar 6.054783e+06 1.926465e+06
  NSGA-II       best_cvar 3.863770e+06 9.305883e+05
    SA-NA       best_cvar 6.872119e+05 3.394566e+05
  SA-only       best_cvar 6.324139e+05 4.443880e+05
    SPEA2       best_cvar 3.787995e+06 1.064467e+06
   MOEA/D              hv 5.294146e-01 1.770425e-01
  NSGA-II              hv 7.955682e-01 1.192299e-01
    SA-NA              hv 8.189922e-01 2.056332e-02
  SA-only              hv 7.908975e-01 9.432485e-02
    SPEA2              hv 7.982646e-01 9.604191e-02
   MOEA/D             igd 4.204520e-01 1.622985e-01
  NSGA-II             igd 2.312417e-01 6.358800e-02
    SA-NA             igd 1.868328e-01 2.051843e-02
  SA-only             igd 2.229010e-01 7.129470e-02
    SPEA2             igd 2.323820e-01 6.906971e-02
```

Pairwise Mann-Whitney tests comparing SA-NA against each baseline:

```text
      comparison metric alternative  u_statistic  p_value
SA-NA vs SA-only     hv     greater         70.0 0.070233
SA-NA vs SA-only    igd        less         30.0 0.070233
SA-NA vs NSGA-II     hv     greater         78.0 0.018818
SA-NA vs NSGA-II    igd        less         18.0 0.008629
 SA-NA vs MOEA/D     hv     greater         98.0 0.000165
 SA-NA vs MOEA/D    igd        less          1.0 0.000123
  SA-NA vs SPEA2     hv     greater         41.0 0.763662
  SA-NA vs SPEA2    igd        less         28.0 0.052055
```

Per-run results:

```text
algorithm  seed  runtime_s  objective_calls  pareto_size  best_raroc    best_cvar       hv      igd  na_triggers  na_evals  accepted_moves
    SA-NA     1   0.043263              800            2    0.866420 9.899208e+05 0.815625 0.181267         11.0     220.0           471.0
    SA-NA     2   0.041641              800            6    0.788705 6.903193e+05 0.793028 0.210479         11.0     220.0           460.0
    SA-NA     3   0.041981              800            1    0.959882 1.112725e+06 0.844407 0.154484         17.0     322.0           388.0
    SA-NA     4   0.038570              800            2    0.918612 4.299799e+05 0.860274 0.153836         13.0     259.0           430.0
    SA-NA     5   0.039506              800            1    0.895319 1.314707e+06 0.811876 0.174258          7.0     140.0           493.0
    SA-NA     6   0.043478              800            3    0.791273 5.675298e+05 0.794848 0.203771         19.0     376.0           351.0
    SA-NA     7   0.039596              800            2    0.797854 5.279066e+05 0.810137 0.200277         10.0     186.0           492.0
    SA-NA     8   0.038973              800            1    0.786623 3.501471e+05 0.813885 0.205481         14.0     266.0           423.0
    SA-NA     9   0.048293              800            1    0.834213 5.734111e+05 0.822190 0.188252         20.0     386.0           330.0
    SA-NA    10   0.036333              800            2    0.809765 3.154725e+05 0.823651 0.196223          9.0     179.0           497.0
  SA-only     1   0.039990              800            2    0.658480 8.349129e+05 0.745319 0.245641          0.0       0.0           617.0
  SA-only     2   0.040089              800            1    0.538369 3.484490e+05 0.719885 0.288009          0.0       0.0           611.0
  SA-only     3   0.039179              800            3    0.612835 1.495616e+06 0.693215 0.293715          0.0       0.0           600.0
  SA-only     4   0.038724              800            2    0.994876 2.834622e+05 0.895911 0.138450          0.0       0.0           611.0
  SA-only     5   0.040775              800            2    1.048549 4.324942e+05 0.909192 0.130045          0.0       0.0           576.0
  SA-only     6   0.040206              800            2    0.792509 1.347454e+06 0.772908 0.210595          0.0       0.0           617.0
  SA-only     7   0.040368              800            1    0.723427 4.841948e+05 0.784394 0.225234          0.0       0.0           607.0
  SA-only     8   0.039625              800            2    0.563148 4.054960e+05 0.727096 0.278970          0.0       0.0           607.0
  SA-only     9   0.040723              800            2    1.158302 3.514303e+05 0.954651 0.118225          0.0       0.0           608.0
  SA-only    10   0.037911              800            2    0.502027 3.406301e+05 0.706403 0.300125          0.0       0.0           608.0
  NSGA-II     1   0.068501              800            6    1.152024 4.266835e+06 0.743399 0.258535          NaN       NaN             NaN
  NSGA-II     2   0.084037              800            2    1.210974 3.160657e+06 0.828648 0.173649          NaN       NaN             NaN
  NSGA-II     3   0.068622              800            5    1.120099 4.337851e+06 0.724147 0.264614          NaN       NaN             NaN
  NSGA-II     4   0.066349              800            7    0.910581 4.314763e+06 0.670158 0.316598          NaN       NaN             NaN
  NSGA-II     5   0.067722              800            2    1.330831 4.104437e+06 0.800272 0.242675          NaN       NaN             NaN
  NSGA-II     6   0.065796              800            3    1.410594 4.935713e+06 0.796333 0.234845          NaN       NaN             NaN
  NSGA-II     7   0.067036              800            4    1.175199 4.090776e+06 0.749565 0.262620          NaN       NaN             NaN
  NSGA-II     8   0.067005              800            3    1.820935 1.618494e+06 1.110257 0.082274          NaN       NaN             NaN
  NSGA-II     9   0.066219              800            3    1.102599 3.453560e+06 0.751720 0.254276          NaN       NaN             NaN
  NSGA-II    10   0.066016              800            3    1.259995 4.354613e+06 0.781181 0.222331          NaN       NaN             NaN
   MOEA/D     1   0.173911              800           57    0.912050 7.308674e+06 0.543012 0.413669          NaN       NaN             NaN
   MOEA/D     2   0.168131              800           59    1.017524 4.641207e+06 0.697756 0.261234          NaN       NaN             NaN
   MOEA/D     3   0.161660              800           28    0.767090 6.561643e+06 0.541148 0.394300          NaN       NaN             NaN
   MOEA/D     4   0.167968              800           64    0.503235 8.141321e+06 0.418054 0.501492          NaN       NaN             NaN
   MOEA/D     5   0.167101              800           35    0.680162 8.081441e+06 0.460324 0.461787          NaN       NaN             NaN
   MOEA/D     6   0.164765              800           69    0.179409 8.273035e+06 0.330990 0.605734          NaN       NaN             NaN
   MOEA/D     7   0.163617              800          100    0.133002 4.730440e+06 0.221538 0.734539          NaN       NaN             NaN
   MOEA/D     8   0.162575              800           30    1.344986 4.361275e+06 0.809838 0.206295          NaN       NaN             NaN
   MOEA/D     9   0.163562              800           20    0.614598 2.663958e+06 0.656373 0.294545          NaN       NaN             NaN
   MOEA/D    10   0.161192              800           43    0.912118 5.784839e+06 0.615113 0.330924          NaN       NaN             NaN
    SPEA2     1   0.076424              800            5    1.360750 3.422285e+06 0.840010 0.209725          NaN       NaN             NaN
    SPEA2     2   0.076700              800            4    1.459938 3.895676e+06 0.847124 0.216402          NaN       NaN             NaN
    SPEA2     3   0.076072              800            3    1.121078 6.326218e+06 0.634382 0.361976          NaN       NaN             NaN
    SPEA2     4   0.075531              800            3    0.931178 3.834861e+06 0.698210 0.282641          NaN       NaN             NaN
    SPEA2     5   0.074382              800            3    1.171041 2.999418e+06 0.819667 0.182516          NaN       NaN             NaN
    SPEA2     6   0.074479              800            5    1.449651 3.035632e+06 0.888567 0.184935          NaN       NaN             NaN
    SPEA2     7   0.073974              800            7    1.387957 2.641015e+06 0.895676 0.163340          NaN       NaN             NaN
    SPEA2     8   0.074643              800            5    1.260298 3.417350e+06 0.817381 0.211517          NaN       NaN             NaN
    SPEA2     9   0.076447              800            7    1.023291 4.745059e+06 0.665829 0.332353          NaN       NaN             NaN
    SPEA2    10   0.074903              800            4    1.467856 3.562436e+06 0.875799 0.178416          NaN       NaN             NaN
```
