======================================================================
Query Benchmark Results (Gemini: gemini-3-flash-preview)
======================================================================
Query: 쿼리를 보내는데도 세션이 표시가 안돼. 어떻게 해야 해?
Runs per mode: 1
----------------------------------------------------------------------
Mode       Avg (s)    Min (s)    Max (s)    StdDev     Resp Len  
----------------------------------------------------------------------
naive      12.753     12.753     12.753     0.000      1763      
local      19.629     19.629     19.629     0.000      1466      
global     21.268     21.268     21.268     0.000      1710      
hybrid     18.355     18.355     18.355     0.000      1326      
mix        24.069     24.069     24.069     0.000      1689      
----------------------------------------------------------------------

Fastest: naive (12.753s)
Slowest: mix (24.069s)
Difference: 11.316s (1.9x)
INFO: Closed PostgreSQL database connection pool
INFO: Successfully finalized 12 storages
INFO: Successfully finalized all RAGAnything storages