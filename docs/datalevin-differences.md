# Datalevin vs Datomic/DataScript: Critical Differences

This document captures the differences that **matter for query generation**. The NLQ model must generate valid Datalevin queries, not generic Datalog.

## Quick Reference

| Feature | Datalevin | Datomic | DataScript |
|---------|-----------|---------|------------|
| Schema format | Map of maps | Vector of maps | Map of maps |
| Schema storage | Passed at connection | Transacted | Passed at creation |
| History/time-travel | No | Yes | No |
| Transaction IDs stored | No | Yes | Yes |
| `d/history` function | No | Yes | No |
| `:db/txInstant` | No | Yes | No |
| Full-text search | Built-in `fulltext` | `fulltext` (different) | No |
| AVE index | Always on | Must enable | Must enable |
| AEV index | No | Yes | Yes |
| VEA index | Only for refs | Yes | Yes |
| Supported types | Most (no bigint, bigdec, uri, tuple) | All | Limited |

## Schema Differences

### Datalevin Schema Format
```clojure
;; Map of maps - passed when getting connection
(def schema
  {:user/name    {:db/valueType :db.type/string}
   :user/email   {:db/valueType :db.type/string
                  :db/unique :db.unique/identity}
   :user/age     {:db/valueType :db.type/long}
   :user/friends {:db/valueType :db.type/ref
                  :db/cardinality :db.cardinality/many}})

(d/get-conn "/path/to/db" schema)
```

### Datomic Schema Format
```clojure
;; Vector of maps - transacted into database
(def schema
  [{:db/ident :user/name
    :db/valueType :db.type/string
    :db/cardinality :db.cardinality/one}
   {:db/ident :user/email
    :db/valueType :db.type/string
    :db/cardinality :db.cardinality/one
    :db/unique :db.unique/identity}])

(d/transact conn {:tx-data schema})
```

**Impact on NLQ**: Schema structure doesn't affect queries directly, but the model needs to understand Datalevin's schema when reading it.

## No History / Time-Travel

### Datalevin: Data is mutable
```clojure
;; When you retract data, it's gone
(d/transact! conn [[:db/retract 42 :user/name "Old Name"]])
;; Cannot query historical values
```

### Datomic: Immutable append-only
```clojure
;; Query at a specific point in time
(d/q '[:find ?name :where [?e :user/name ?name]]
     (d/as-of db #inst "2024-01-01"))

;; Get history of changes
(d/q '[:find ?name ?tx ?added
       :where [?e :user/name ?name ?tx ?added]]
     (d/history db))
```

**Impact on NLQ**: The model should **never** generate:
- `d/as-of`
- `d/since`
- `d/history`
- Queries with `?tx` or `?added` pattern positions
- `:db/txInstant` attribute references

## Transaction ID Differences

### Datomic: 5-tuple datoms
```clojure
;; Datomic datoms have 5 positions: [e a v tx added]
[:find ?e ?name ?tx
 :where
 [?e :user/name ?name ?tx]]  ;; Can bind tx
```

### Datalevin: 3-tuple datoms (EAV only)
```clojure
;; Datalevin datoms have 3 positions: [e a v]
[:find ?e ?name
 :where
 [?e :user/name ?name]]  ;; No tx position
```

**Impact on NLQ**: The model should **never** generate patterns with 4+ positions.

## Full-Text Search

### Datalevin: Built-in `fulltext` function
```clojure
;; Schema: enable fulltext
{:article/body {:db/valueType :db.type/string
                :db/fulltext true}}

;; Query: use fulltext function
(d/q '[:find ?e ?a ?v
       :in $ ?q
       :where [(fulltext $ ?q) [[?e ?a ?v]]]]
     db "search terms")

;; Attribute-specific search
(d/q '[:find ?e ?a ?v
       :in $ ?q
       :where [(fulltext $ :article/body ?q) [[?e ?a ?v]]]]
     db "search terms")

;; Boolean query syntax
(d/q '[:find ?e ?a ?v
       :where [(fulltext $ [:and "red" [:not "blue"]]) [[?e ?a ?v]]]]
     db)

;; Phrase search
(d/q '[:find ?e ?a ?v
       :where [(fulltext $ {:phrase "exact phrase"}) [[?e ?a ?v]]]]
     db)
```

### Datomic: Different `fulltext` function
```clojure
;; Datomic fulltext searches single attribute
(d/q '[:find ?e ?name
       :where [(fulltext $ :user/bio "search term") [[?e ?name]]]]
     db)
```

**Impact on NLQ**: Must use Datalevin's fulltext syntax, especially:
- `[[?e ?a ?v]]` destructuring pattern
- Boolean operators `:and`, `:or`, `:not`
- `{:phrase "..."}` for phrase search

## Index Differences

### Datalevin indexes
- **EAVT**: Entity-Attribute-Value (always)
- **AEVT**: Attribute-Entity-Value (always)
- **AVE**: Attribute-Value-Entity (always on, unlike Datomic)
- **VEA**: Value-Entity-Attribute (only for `:db.type/ref` attributes)

### Datomic indexes
- EAVT, AEVT, VAET (always)
- AVE (must set `:db/index true`)

**Impact on NLQ**: Datalevin's always-on AVE index means value-based queries are efficient without needing `:db/index true`. The model doesn't need to worry about index availability.

## Supported Value Types

### Datalevin supports:
- `:db.type/string`
- `:db.type/long`
- `:db.type/double`
- `:db.type/float`
- `:db.type/boolean`
- `:db.type/instant`
- `:db.type/uuid`
- `:db.type/ref`
- `:db.type/bytes`
- `:db.type/keyword`
- `:db.type/symbol`

### Datalevin does NOT support:
- `:db.type/bigint`
- `:db.type/bigdec`
- `:db.type/uri`
- `:db.type/tuple`

**Impact on NLQ**: Avoid generating queries that assume these unsupported types exist.

## Built-in Functions and Predicates

### Common to both (safe to use):
```clojure
;; Comparison
[(< ?age 30)]
[(<= ?age 30)]
[(> ?age 30)]
[(>= ?age 30)]
[(= ?status :active)]
[(not= ?status :deleted)]

;; Math
[(+ ?a ?b) ?sum]
[(- ?a ?b) ?diff]
[(* ?a ?b) ?product]
[(/ ?a ?b) ?quotient]
[(mod ?a ?b) ?remainder]

;; String operations (via clojure.string)
[(clojure.string/includes? ?email "gmail")]
[(clojure.string/starts-with? ?name "Dr.")]
[(clojure.string/ends-with? ?file ".pdf")]
[(clojure.string/lower-case ?name) ?lower]
[(clojure.string/upper-case ?name) ?upper]

;; Type predicates
[(string? ?v)]
[(number? ?v)]
[(keyword? ?v)]

;; Collection operations
[(count ?items) ?n]
[(contains? ?tags :important)]

;; Missing attribute check
[(missing? $ ?e :user/deleted)]

;; Get attribute (for dynamic access)
[(get-else $ ?e :user/nickname "Anonymous") ?nick]
```

### Datalevin-specific:
```clojure
;; Full-text search
[(fulltext $ ?query) [[?e ?a ?v]]]
[(fulltext $ :attr ?query) [[?e ?a ?v]]]
[(fulltext $ :attr ?query {:top 10}) [[?e ?a ?v]]]

;; Java interop (Datalevin allows this)
[(.getTime ?date) ?timestamp]
[(.after ?date1 ?date2)]
[(.toLowerCase ?str) ?lower]
```

### Aggregation functions:
```clojure
;; These work in both
[:find (count ?e) :where [?e :user/name _]]
[:find (sum ?amount) :where [?e :order/amount ?amount]]
[:find (avg ?age) :where [?e :user/age ?age]]
[:find (min ?date) :where [?e :created-at ?date]]
[:find (max ?price) :where [?e :product/price ?price]]
[:find (count-distinct ?category) :where [?e :product/category ?category]]
[:find (sample 5 ?name) :where [?e :user/name ?name]]
[:find (median ?age) :where [?e :user/age ?age]]  ;; Datalevin has this!
[(variance ?ages)]  ;; Datalevin has this!
[(stddev ?ages)]    ;; Datalevin has this!
```

## Query Syntax Compatibility

### Standard patterns (work in both):
```clojure
;; Basic pattern
[:find ?e ?name
 :where [?e :user/name ?name]]

;; With inputs
[:find ?e
 :in $ ?name
 :where [?e :user/name ?name]]

;; Multiple inputs
[:find ?e
 :in $ ?name ?age
 :where
 [?e :user/name ?name]
 [?e :user/age ?age]]

;; Collection input
[:find ?e
 :in $ [?name ...]
 :where [?e :user/name ?name]]

;; Tuple input
[:find ?e
 :in $ [?min ?max]
 :where
 [?e :user/age ?age]
 [(>= ?age ?min)]
 [(<= ?age ?max)]]

;; Or clauses
[:find ?e
 :where
 (or [?e :user/status :active]
     [?e :user/status :pending])]

;; And clauses (implicit, but can be explicit)
[:find ?e
 :where
 (and [?e :user/name ?name]
      [?e :user/active true])]

;; Not clauses
[:find ?e
 :where
 [?e :user/name ?name]
 (not [?e :user/deleted true])]

;; Or-join
[:find ?e
 :where
 (or-join [?e]
   [?e :user/email ?email]
   [?e :user/phone ?phone])]

;; Not-join
[:find ?e
 :where
 [?e :user/name ?name]
 (not-join [?e]
   [?e :user/role :admin]
   [?e :user/role :superuser])]

;; Rules
[:find ?e
 :in $ %
 :where (ancestor ?e ?parent)]

;; Pull syntax
[:find (pull ?e [:user/name :user/email])
 :where [?e :user/active true]]

;; Pull with nested refs
[:find (pull ?e [:user/name {:user/orders [:order/total :order/date]}])
 :where [?e :user/active true]]
```

### Return specifications:
```clojure
;; Relation (default) - returns set of tuples
[:find ?e ?name :where ...]

;; Collection - returns flat collection
[:find [?name ...] :where [_ :user/name ?name]]

;; Single tuple
[:find [?e ?name] :where [?e :user/name ?name]]

;; Scalar
[:find ?name . :where [?e :user/name ?name] [?e :user/id 123]]

;; With keys (returns maps)
[:find ?e ?name ?email
 :keys id name email
 :where ...]
```

## Date/Time Handling

```clojure
;; Instant literals
[:find ?e
 :where
 [?e :user/created ?date]
 [(> ?date #inst "2024-01-01T00:00:00Z")]]

;; Java interop for date comparison
[:find ?e
 :where
 [?e :user/created ?date]
 [(.after ?date #inst "2024-01-01")]]

;; Get current time
[:find ?e
 :in $ ?now
 :where
 [?e :user/expires ?exp]
 [(< ?exp ?now)]]
;; Called with: (d/q query db (java.util.Date.))
```

## Rules

```clojure
;; Define rules (same syntax in both)
(def rules
  '[[(ancestor ?e ?a)
     [?e :person/parent ?a]]
    [(ancestor ?e ?a)
     [?e :person/parent ?p]
     (ancestor ?p ?a)]])

;; Use rules
(d/q '[:find ?e ?ancestor
       :in $ %
       :where (ancestor ?e ?ancestor)]
     db rules)
```

## Summary: What to Generate / What to Avoid

### SAFE to generate:
- Standard `:find`, `:where`, `:in` clauses
- All comparison predicates
- `clojure.string/*` functions
- `fulltext` with Datalevin's syntax
- Pull syntax
- Rules
- Aggregations including `median`, `variance`, `stddev`
- Or/Not/And clauses
- Collection and tuple inputs
- Java interop on values

### AVOID generating:
- `d/as-of`, `d/since`, `d/history`
- 4-position patterns `[?e ?a ?v ?tx]`
- `:db/txInstant`
- Datomic-specific fulltext syntax
- References to unsupported types (bigint, uri, tuple)
- `d/entity` in queries (it's an API call, not query syntax)

## Example Transformations

| Natural Language | Datalevin Query |
|-----------------|-----------------|
| "Find all users" | `[:find ?e :where [?e :user/name _]]` |
| "Users named John" | `[:find ?e :where [?e :user/name "John"]]` |
| "Users older than 30" | `[:find ?e :where [?e :user/age ?a] [(> ?a 30)]]` |
| "Users with gmail" | `[:find ?e :where [?e :user/email ?e] [(clojure.string/includes? ?e "gmail")]]` |
| "Count all orders" | `[:find (count ?e) :where [?e :order/id _]]` |
| "Average order total" | `[:find (avg ?t) :where [?e :order/total ?t]]` |
| "Search articles for 'clojure'" | `[:find ?e ?a ?v :where [(fulltext $ "clojure") [[?e ?a ?v]]]]` |
| "Users without email" | `[:find ?e :where [?e :user/name _] (not [?e :user/email _])]` |
| "Orders with their user names" | `[:find ?o ?name :where [?o :order/user ?u] [?u :user/name ?name]]` |

## References

- [Datalevin GitHub](https://github.com/juji-io/datalevin)
- [Datalevin Query Docs](https://github.com/juji-io/datalevin/blob/master/doc/query.md)
- [Datalevin Search Docs](https://github.com/juji-io/datalevin/blob/master/doc/search.md)
- [Datomic Query Reference](https://docs.datomic.com/cloud/query/query-data-reference.html)
