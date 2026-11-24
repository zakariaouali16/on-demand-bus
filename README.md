# On-Demand Bus Dispatcher (Demand-Responsive Transit)

Buses without a fixed line. Customers request a trip; the dispatcher picks a
bus, a **pickup station and drop-off station within walking distance**, and the
**order of stops**, so each bus drives the streets closest to all its riders,
not door-to-door and not a fixed loop. The plan is **re-optimised every minute**
as new requests arrive, so routes keep changing while buses drive.

Buses drive on **real streets from OpenStreetMap** (one-way streets, speed
limits). IBM **watsonx.ai** is used twice:
- **Granite** chat turns free-text bookings like *"me and my sister from the
  Ferry Building to City Hall"* into structured trips.
- **Granite TinyTimeMixer** forecasts requests per zone for the next hour, so
  idle buses wait where customers are about to appear, and drivers take their
  breaks before the rush instead of during it.

The fleet also keeps **every zone within 5 minutes of a bus**, parks spare
buses at the **depot** when it is quiet, and gives drivers **legal breaks**.

## Quick start

```bash
pip install -r requirements.txt

# 1) Benchmark vs. a fixed bus line with the same fleet
python -m drt --buses 8 --rate 60 --hours 2      # add --forecast naive to save watsonx quota

# 2) Live map (click twice on the map to book, or type a booking)
python -m uvicorn drt.server:app
# open http://127.0.0.1:8000

# 3) Tests
python -m unittest -v
```

### Choose the city

Set `DRT_CITY` in `.env` to any place OpenStreetMap knows, and `DRT_RADIUS_M` to
the size of the area served:

```
DRT_CITY=Union Square, San Francisco, California, USA
DRT_RADIUS_M=2000
```

The first start downloads the streets and named places (1-3 min). After that
they load from `data/` in about a second. Named places (stations, hospitals,
universities, parks, museums, malls) are found automatically and can be used in
plain-language bookings. Use `python -m drt --city grid` for the synthetic test
city.

### Enable watsonx

Put the key file downloaded from IBM Cloud in the project folder as
`apikey.json`, and add to `.env`:

```
WATSONX_PROJECT_ID=<watsonx.ai project id>
WATSONX_URL=https://us-south.ml.cloud.ibm.com
WATSONX_MODEL_ID=ibm/granite-4-h-small   # optional
```

The project must have a **watsonx.ai Runtime** service associated (Manage →
Services & integrations). Without watsonx, a rule-based parser is used.
`apikey.json`, `.env` and `data/` are git-ignored.

**Quota:** the Lite plan has a monthly limit. Benchmarks call the forecast every
simulated 15 min; the live server makes at most one call every 2 min
(`WATSONX_FORECAST_MIN_INTERVAL_S`). If the quota runs out, the app logs it,
pauses watsonx for an hour and uses the built-in fallback forecast, so nothing
breaks. Run `python -m drt --forecast naive` to save quota during experiments.
Note that IBM lists the TTM models as deprecated; if they stop working, the
fallback takes over automatically.

## How it works

| Step | What happens | Code |
|---|---|---|
| Streets | OpenStreetMap drivable streets, with one-ways and speed limits. For each station, the travel time and next turn from every intersection are precomputed, so routing lookups are instant. | `drt/network.py` |
| Stations | Virtual stations about 350 m apart, on real intersections (busier ones preferred, landmarks always included). Each customer gets up to 3 candidate stations within `max_walk_m` at both ends. | `drt/stations.py` |
| Insertion | For every bus × pickup station × drop-off station × position in the stop list, simulate the route and keep the cheapest option. Cost = bus time + passenger time + 1.5 × walking time. | `Dispatcher.best_insertion` |
| Promises | Hard limits: capacity, max wait at the station (10 min), max ride (1.8× direct car time, or direct + 5 min). A new customer is refused if accepting them would make an already-booked customer late. | `Dispatcher.plan` |
| Re-optimise | Every 60 s, all not-yet-boarded customers are removed and re-inserted, possibly on another bus or with a new drop-off station. The new plan is kept only if it is cheaper. A pickup station is never changed once given, because the customer is already walking there. | `Dispatcher.reoptimize` |
| Zones + demand | Stations are grouped into 8 zones. Demand follows the time of day: the downtown core attracts trips in the morning, residential zones in the evening, with peaks at 08:00 and 17:30. 6 days of history are generated for the forecaster. | `drt/demand.py` |
| Forecast | Every 15 min, the last 512 slots per zone go to watsonx `time_series/forecast` (`ibm/granite-ttm-512-96-r2`), which predicts the next hour. The built-in fallback forecast (same slot on previous days, scaled to today) is used if watsonx is unavailable. Both are scored against what actually happens. | `drt/forecast.py` |
| Coverage | Every zone must have a bus within 5 min of its centre. Uncovered zones get the nearest idle bus that isn't the only cover for another zone. | `FleetManager._position_idle` |
| Forecast placement | Each zone should have buses in proportion to its forecast. Idle buses in over-supplied zones move to under-supplied zones within 10 min; the others stay put. | `FleetManager._position_idle` |
| Depot | When the forecast is quiet everywhere, spare buses wait at the depot. They can still take bookings from there. | `FleetManager` |
| Breaks | After 3 h, a driver takes no new customers, finishes the current ones, drives to the depot and rests 30 min. At most 2 buses are off at once. Breaks are taken early (after at least 1.5 h) when it's quiet or a rush is coming, and postponed (up to 30 min) during a rush that's about to ease. A driver past the hard limit always rests. | `FleetManager._manage_breaks` |
| Booking | watsonx.ai extracts origin, destination and passenger count from text. | `drt/watsonx.py` |

## Benchmark results

### Forecasting (San Francisco, 8 buses, 2 h from 07:00, average of 3 runs)

| Average demand | | Served | Wait at stop | Door-to-door |
|---|---|---|---|---|
| 60/h | no forecast | 96.2% | 5.11 min | 16.99 min |
| | **forecast** | 95.2% | **4.84 min** | **16.54 min** |
| 100/h | no forecast | 69.4% | 6.77 min | 20.36 min |
| | **forecast** | **74.3%** | **6.30 min** | **19.86 min** |

"No forecast" still keeps zone coverage and breaks, but positions buses and
times breaks without knowing what's coming. At 100/h the morning peak is about
220 requests/h, which is more than 8 buses can carry.

Forecast accuracy (average error per zone per 15 min): watsonx TTM 2.48, built-in
fallback 2.29. The synthetic demand repeats exactly every day, which suits the
fallback. Real ridership, with trends, weather and events, is where a learned model
should pull ahead.

### Earlier results (flat demand, before zones and breaks)

Downtown San Francisco (2 km radius: 1,575 intersections, 89 stations), 8 buses × 12 seats, 150 requests/h:

| | On-demand | Fixed line |
|---|---|---|
| Door-to-door trip | **16.5 min** | 29.4 min |
| Walking | 454 m | 1,441 m |
| Wait at station | 4.4 min | 8.7 min |
| Requests served | 99% | – |
| On-demand faster for | 97% of trips | |

Synthetic 3.3 × 3.3 km grid city, same fleet:

| | 150 requests/h | 400 requests/h |
|---|---|---|
| Door-to-door, on-demand | **16.1 min** | **20.9 min** |
| Door-to-door, fixed line | 29.0 min | 28.8 min |
| Wait at station, on-demand / fixed | 4.3 / 8.7 min | 6.3 / 8.7 min |
| Requests served | 100% | 60% (fleet saturated) |
| On-demand faster for | 99% of trips | 80% of trips |

The fixed-line baseline ignores capacity, which makes the baseline look better
than it would be in practice.

## Project layout

```
drt/
  network.py     OpenStreetMap streets (or test grid), precomputed routing
  stations.py    virtual stations + walking candidates
  models.py      Config, Request, Bus, Stop
  dispatcher.py  insertion + re-optimisation (the core algorithm)
  demand.py      zones + time-of-day demand model + history
  forecast.py    watsonx TTM forecasting (+ fallback, accuracy tracking)
  fleet.py       coverage, forecast placement, depot, driver breaks
  simulation.py  demand, bus movement, metrics, fixed-line baseline
  watsonx.py     watsonx.ai natural-language booking
  server.py      FastAPI live API
  static/index.html  live map
tests/test_drt.py
```

## Next steps

- **Book any address**: geocode free-text addresses (e.g. with `osmnx.geocode`)
  so riders are not limited to the named places.
- **Real ridership data**: replace the synthetic history in `demand.py` with
  real trip records to get the most out of the watsonx forecast.
- **Live traffic**: update the edge `time` values from traffic feeds; the
  re-optimiser adapts automatically.
- **Scale**: for large fleets, limit insertion to the K nearest buses, or replace
  the insertion step with OR-Tools or a dial-a-ride solver.
- **IBM Bob**: open this folder in IBM Bob to extend it. Point it at
  `dispatcher.py` and the tests and ask for features such as time windows for
  scheduled trips or wheelchair seats.
