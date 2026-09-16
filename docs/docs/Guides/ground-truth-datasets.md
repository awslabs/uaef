# Ground Truth Dataset

One row per test case, `.csv` or `.xlsx`.

| Column | Required | Also accepted as | Meaning |
|---|---|---|---|
| `Question` | yes | `Query`, `Input` | What to send the agent |
| `Answer` | no | `Expected_Answer`, `Expected_Output`, `Expected` | What it should have said |
| `context` | no | — | Reference documents, for retrieval-style evaluation |
| `expected_tool_calls` | no | `tools` | A JSON array, e.g. `[{"name": "get_weather", "arguments": {"location": "Seattle"}}]` |
| `session_id` | only for multi-turn | — | Groups rows into one conversation |
| `turn_id` | only for multi-turn | — | Orders the turns within a conversation |

Only `Question` is mandatory. Without `Answer` you still get safety, performance, and
reasoning scores; with it you also get the comparison metrics such as `accuracy`. The
pre-built files for each lab already have the right columns, so you will not need to
build one today.

## When you need `session_id` and `turn_id`

Leave both out and every row is an independent test case — the agent is asked the
question with no memory of anything before it. That is Lab 1, and it is the right shape
for most evaluation.

Add them when the thing you want to measure is the *conversation* rather than the
individual answers. Once rows share a `session_id`, the service stops treating them as
separate tests: it drives them in order against a single agent session, so a turn can
depend on what came before. That is what makes a question like "Book the cheapest
hotel." meaningful — on its own it has no referent, and it only scores well if the agent
remembers the list it just gave you.

This is also what unlocks the Multi-Turn dimension. Without `session_id` there is no
conversation to score, so metrics like context retention and conversation completeness
have nothing to work with.

### Specifics worth knowing

- **`turn_id` is technically optional.** If it is absent, turns run in the order the
  rows appear in the file. Include it anyway — sorting or editing a spreadsheet
  reorders rows silently, and a scrambled conversation scores badly for reasons that
  have nothing to do with the agent.
- **A row with no `session_id` becomes a conversation of one.** So a single file can
  hold both single-turn rows and multi-turn conversations, and the single-turn rows
  behave exactly as they did in Lab 1.
- **`Answer` and `expected_tool_calls` are per row, meaning per turn.** In a multi-turn
  file each turn carries its own expected answer and expected tools, which is what
  produces the per-turn breakdown in the results.
- **These two column names are read literally, in lowercase.** Unlike `Question` and
  `Answer`, which accept any capitalisation, `session_id` and `turn_id` must be spelled
  exactly that way. A column called `Session_ID` is not recognised, and the file will
  quietly be evaluated as single-turn.
- **Multi-turn needs an endpoint that keeps conversation state.** The service sends each
  turn with the session id and does not replay earlier turns, so the agent has to
  remember them itself. The lab endpoint does; a plain request/response wrapper would
  not.

## Checking you got it right

After uploading, click **Validate**. The result line reports the shape the run will
actually use:

```
MULTI-TURN — 9 rows, 2 conversations (grouped by session_id)
```

If that says `SINGLE-TURN` when you expected conversations, the `session_id` column is
missing or misspelled. This is worth checking before running — a multi-turn dataset
evaluated as single-turn produces a complete set of plausible scores, with nothing to
indicate the conversations were never assembled.

## Examples

Single-turn, the Lab 1 shape:

```csv
Question,Answer,expected_tool_calls
What time is my upcoming flight?,"LX0112 CDG→BSL at 12:09 PM","[{""name"": ""fetch_user_flight_information"", ""arguments"": {}}]"
Any museums in Basel?,Kunstmuseum Basel,"[{""name"": ""search_trip_recommendations"", ""arguments"": {""location"": ""Basel""}}]"
```

Multi-turn, the Lab 2 shape — the same two columns plus `session_id` and `turn_id`,
where turn 2 only makes sense after turn 1:

```csv
session_id,turn_id,Question,Answer,expected_tool_calls
session_001,1,I like a cheap hotel for my 7 day stay.,"Cheapest first: Holiday Inn (ID 8), Hyatt (ID 3), Hilton (ID 1)","[{""name"": ""search_hotels"", ""arguments"": {""location"": ""Basel""}}]"
session_001,2,Book the cheapest hotel.,"Holiday Inn Basel booked","[{""name"": ""book_hotel"", ""arguments"": {""hotel_id"": 8}}]"
```
