# Single Agent Notebook Specification

## Agent: Weekly Weather Assistant

A single agent with 5 tools providing 7-day forecast data (Monday–Sunday). Each query exercises one or more tools.

## Tools

| Tool | Input | Returns |
|------|-------|---------|
| `get_temperature(day)` | Day of week | `{"day": "...", "temperature_f": int}` |
| `get_air_quality(day)` | Day of week | `{"day": "...", "air_quality_aqi": int}` |
| `get_weather_condition(day)` | Day of week | `{"day": "...", "condition": "sunny"|"cloudy"|"rain"}` |
| `get_humidity(day)` | Day of week | `{"day": "...", "humidity_pct": int}` |
| `get_wind_speed(day)` | Day of week | `{"day": "...", "wind_speed_mph": int}` |

## Data (hardcoded)

| Day | Temp (°F) | AQI | Condition | Humidity (%) | Wind (mph) |
|-----|-----------|-----|-----------|--------------|------------|
| Monday | 58 | 42 | sunny | 45 | 8 |
| Tuesday | 63 | 55 | cloudy | 62 | 12 |
| Wednesday | 71 | 38 | sunny | 40 | 5 |
| Thursday | 55 | 72 | rain | 85 | 18 |
| Friday | 49 | 65 | rain | 90 | 22 |
| Saturday | 67 | 30 | sunny | 38 | 6 |
| Sunday | 74 | 28 | sunny | 35 | 4 |

## System Prompt

> You are a weekly weather assistant. You have access to 7-day forecast data (Monday through Sunday). Use the appropriate tool to answer questions about temperature, air quality, weather conditions, humidity, or wind speed for any day of the week. Always call the relevant tool before answering.

## Test Cases (7 queries)

| # | Query | Expected Tool(s) | Expected Output |
|---|-------|-------------------|-----------------|
| 1 | "What is the temperature on Wednesday?" | `get_temperature(day="Wednesday")` | "The temperature on Wednesday is 71°F." |
| 2 | "How is the air quality on Thursday?" | `get_air_quality(day="Thursday")` | "The air quality index on Thursday is 72 AQI, which is moderate." |
| 3 | "Will it rain on Friday?" | `get_weather_condition(day="Friday")` | "Yes, the weather condition on Friday is rain." |
| 4 | "What is the humidity on Tuesday?" | `get_humidity(day="Tuesday")` | "The humidity on Tuesday is 62%." |
| 5 | "How windy is it on Friday?" | `get_wind_speed(day="Friday")` | "The wind speed on Friday is 22 mph." |
| 6 | "What's the weather like on Saturday?" | `get_weather_condition(day="Saturday")` + `get_temperature(day="Saturday")` | "Saturday is sunny with a temperature of 67°F, humidity at 38%, and wind speed of 6 mph." |
| 7 | "Is Sunday a good day for outdoor activities?" | `get_weather_condition(day="Sunday")` + `get_temperature(day="Sunday")` + `get_air_quality(day="Sunday")` | "Sunday is excellent for outdoor activities: sunny, 74°F, low humidity (35%), light wind (4 mph), and great air quality (AQI 28)." |

## Evaluation

- Use `single_ag_metrics` from `metrics.py` (tool calling + response quality + responsible AI + performance + multi-turn + reasoning).
- Run `batch_evaluate()` over all 7 traces with their ground truths.
- Report per-query score, average score, and pass rate.

## Notebook Structure

1. Auth & imports (dotenv, UAEF, metrics)
2. Define tools + hardcoded data + create agent
3. Define test cases with expected outputs and expected tool calls
4. Loop: run agent on each query → adapter → collect traces + ground truths
5. Select metrics → `batch_evaluate()` → print results


---

## Agent: Single Agent: Booking Assistant

A travel booking assistant backed by a live SQLite database (`travel2.sqlite`). The agent manages flights, hotels, car rentals, and excursions for a given passenger via 16 tools. Uses `ChatBedrockConverse` (`us.anthropic.claude-sonnet-4-6`) with no checkpointer (stateless per invocation). Passenger ID: `3442 587242`.

## Tools

| Tool | Input | Returns |
|------|-------|---------|
| `fetch_user_flight_information()` | *(from config via `InjectedToolArg`)* passenger_id | List of ticket + flight records for the passenger |
| `search_flights(departure_airport, arrival_airport, start_time, end_time, limit)` | Airport codes, date range, limit (default 20) | Available flights |
| `update_ticket_to_new_flight(ticket_no, new_flight_id)` | Ticket number, new flight ID | Confirmation of update; rejects if < 3 hours to departure |
| `cancel_ticket(ticket_no)` | Ticket number | Cancellation confirmation |
| `search_car_rentals(location, name, price_tier, start_date, end_date)` | Location + filters (all optional) | Available car rentals |
| `book_car_rental(rental_id)` | Rental ID | Booking confirmation |
| `update_car_rental(rental_id, start_date, end_date)` | Rental ID, new dates | Update confirmation |
| `cancel_car_rental(rental_id)` | Rental ID | Cancellation confirmation |
| `search_hotels(location, name, price_tier, checkin_date, checkout_date)` | Location + filters (all optional) | Available hotels |
| `book_hotel(hotel_id)` | Hotel ID | Booking confirmation |
| `update_hotel(hotel_id, checkin_date, checkout_date)` | Hotel ID, new dates | Update confirmation |
| `cancel_hotel(hotel_id)` | Hotel ID | Cancellation confirmation |
| `search_trip_recommendations(location, name, keywords)` | Location, name, keywords (all optional) | Recommended excursions/attractions |
| `book_excursion(recommendation_id)` | Recommendation ID | Booking confirmation |
| `update_excursion(recommendation_id, details)` | Recommendation ID, new details string | Update confirmation |
| `cancel_excursion(recommendation_id)` | Recommendation ID | Cancellation confirmation |

## Database Setup

- `ensure_db()`: download `travel2.sqlite` from GCS to `/tmp/travel2.backup.sqlite` on cold start (no-op on warm start).
- `reset_db(eval_mode)`: restore a clean DB before each invocation.
  - `eval_mode=True` (default): copy backup as-is — frozen, reproducible world; pair with `_get_db_reference_time()` as `current_time`.
  - `eval_mode=False`: copy backup then shift all dates to now (production behaviour).
- `_get_db_reference_time()`: returns `max(actual_departure)` from the raw backup as a stable time anchor for eval runs.

## System Prompt

> You are a helpful customer support assistant for Swiss Airlines. Use the provided tools to search for flights, company policies, and other information to assist the user's queries. When searching, be persistent. Expand your query bounds if the first search returns no results. If a search comes up empty, expand your search before giving up.
>
> Current user: `{user_info}`  
> Current time: `{time}`.

## Test Cases (6 queries — passenger `3442 587242`)

| # | Query | Expected Tool(s) | Expected Output |
|---|-------|-------------------|-----------------|
| 1 | "Hi there, what time is my flight?" | `fetch_user_flight_information()` | "Your flight number is LX0112 from Paris (CDG) to Basel (BSL), departing at 6:59 PM today." |
| 2 | "Update my flight and book it for first flight next week" | `search_flights(departure_airport="CDG", arrival_airport="BSL", start_time="2024-11-11", end_time="2024-11-17")` + `update_ticket_to_new_flight(ticket_no="7240005432906569", new_flight_id="19251")` | "I have updated your ticket 7240005432906569 to flight LX0112 departing CDG→BSL on November 12, 2024 at 10:00 PM." |
| 3 | "I like a cheap hotel for my 7 days stay. Provide options sorted cheapest to most expensive." | `search_hotels(location="Basel", price_tier="Midscale", checkin_date="2024-11-08", checkout_date="2024-11-15")` + `search_hotels(..., price_tier="Upper Midscale")` | Lists available hotels in Basel sorted by price tier |
| 4 | "Book the cheapest hotel." | `book_hotel(hotel_id="1")` | "Your hotel confirmation number is HB7712." |
| 5 | "Cool so now what recommendations do you have for the museum?" | `search_trip_recommendations(location="Basel", keywords="museum")` | Recommends Kunstmuseum Basel |
| 6 | "OK great pick one and book it for my second day there." | `book_excursion(recommendation_id="2")` | "Booked Kunstmuseum Basel for Friday, November 9th." |

## Evaluation

- Use `single_ag_metrics` from `metrics.py`.
- Each query runs in an **independent** session (fresh `thread_id` per query, no checkpointer) so there is no cross-turn state dependency in the batch run.
- Call `reset_db()` before each query to restore a clean DB state.
- Run `batch_evaluate()` over all 6 traces with their ground truths.
- Export results to `evaluation-results/batch-eval/booking_assistant_eval.csv`.
- Report per-query score, average score, and pass rate.

## Notebook Structure

1. Auth & imports (dotenv, UAEF, metrics, LangChain, Bedrock)
2. Step 1 — Set up database: `ensure_db()`, `_get_db_reference_time()`, `reset_db(eval_mode=True)`
3. Step 2 — Define 16 booking tools + build LangGraph (`ChatBedrockConverse` `us.anthropic.claude-sonnet-4-6`, `Assistant` re-prompt class, `ToolNode` with error fallback, no checkpointer)
4. Step 3 — Load ground truth from `data/ground-truth-booking.xlsx`; define `TEST_CASES` with `query`, `expected_output`, `expected_tool_calls`
5. Step 4 — Loop: `reset_db()` → run agent with `config={"configurable": {"passenger_id": ..., "current_time": eval_time}}` → `LangGraphAdapter(agent_node_name="assistant")` → collect `traces` + `ground_truths`
6. Step 5 — `batch_evaluate()` → tabular results + export CSV
