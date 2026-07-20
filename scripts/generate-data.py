#!/usr/bin/env python3
"""
Generate training data for Datalevin NLQ model.

Produces JSONL files in the format expected by MLX fine-tuning, wrapped in the
base model's chat template (see CHAT_TEMPLATES / TEMPLATE below):
{"text": "<|im_start|>user\nSchema: {...}\n\nQuery<|im_end|>..."}

Design notes
------------
The model sees the schema in its prompt and must generalize to schemas it has
never seen. That makes attribute-copying -- lifting the right `:ns/attr` out of
the prompt -- the core skill, and *schema diversity* the binding constraint on
whether it is learned rather than memorized.

Two consequences shape this file:

1. Examples are generated programmatically from typed schema definitions, not
   hand-written per schema. Adding a schema costs ~10 lines and yields ~60
   examples across every query pattern.

2. The train/valid split is schema-disjoint (see HOLDOUT). Validation schemas
   never appear in training, so validation loss measures generalization to an
   unseen schema, which is the deployment condition.
"""

import json
import random
import re
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, field

OUTPUT_DIR = Path(__file__).parent.parent / "training-data"
OUTPUT_DIR.mkdir(exist_ok=True)

TRAIN_FILE = OUTPUT_DIR / "train.jsonl"
VALID_FILE = OUTPUT_DIR / "valid.jsonl"

SEED = 42

# Schemas held out entirely for validation. Chosen to span naming conventions
# (abbreviated, kebab-case, plain) rather than to be a random sample.
HOLDOUT = {"clinic", "airline", "podcast", "parking", "bugtracker"}


# --------------------------------------------------------------------------
# Schema model
# --------------------------------------------------------------------------

@dataclass
class Attr:
    name: str                            # "reorder-point" (bare, no namespace)
    type: str                            # string|long|double|instant|boolean|keyword|ref
    values: List[str] = field(default_factory=list)   # EDN literals for equality
    nums: List[str] = field(default_factory=list)     # EDN numbers for comparison
    subs: List[str] = field(default_factory=list)     # substrings for string ops
    unique: bool = False
    many: bool = False
    fulltext: bool = False
    ref_to: Optional[str] = None         # entity namespace this ref points at
    label: Optional[str] = None          # human wording; defaults from name

    def words(self) -> str:
        if self.label:
            return self.label
        return DEFAULT_LABELS.get(self.name, self.name.replace("-", " "))

    @property
    def participle(self) -> bool:
        """True when the attribute name reads as a past participle, so it can be
        phrased directly ("posts published after 2024") rather than needing a
        noun frame ("permits with expiry date after 2024")."""
        return self.name.endswith("ed") and self.type == "instant"

    def substrings(self) -> List[str]:
        return self.subs or DEFAULT_SUBS.get(self.name, [])

    @property
    def numeric(self) -> bool:
        return self.type in ("long", "double")


@dataclass
class Entity:
    ns: str                              # "user"
    noun: str                            # "user"
    plural: str                          # "users"
    primary: str                         # attr name used for "find all X"
    attrs: List[Attr]

    def kw(self, attr_name: str) -> str:
        return f":{self.ns}/{attr_name}"

    @property
    def primary_kw(self) -> str:
        return self.kw(self.primary)

    def by_type(self, *types) -> List[Attr]:
        return [a for a in self.attrs if a.type in types]

    def scalars(self) -> List[Attr]:
        return [a for a in self.attrs if a.type != "ref"]


@dataclass
class Schema:
    name: str
    description: str
    entities: List[Entity]

    @property
    def edn(self) -> str:
        lines = []
        for e in self.entities:
            for a in e.attrs:
                opts = [f":db/valueType :db.type/{a.type}"]
                if a.unique:
                    opts.append(":db/unique :db.unique/identity")
                if a.many:
                    opts.append(":db/cardinality :db.cardinality/many")
                if a.fulltext:
                    opts.append(":db/fulltext true")
                lines.append(f"{e.kw(a.name)} {{{' '.join(opts)}}}")
        body = "\n ".join(lines)
        return "{" + body + "}"

    def entity(self, ns: str) -> Optional[Entity]:
        return next((e for e in self.entities if e.ns == ns), None)

    @property
    def attr_kws(self) -> set:
        return {e.kw(a.name) for e in self.entities for a in e.attrs}


@dataclass
class Example:
    schema: Schema
    natural: str
    query: str
    category: str


def A(name, type_, **kw) -> Attr:
    return Attr(name=name, type=type_, **kw)


def E(ns, noun, plural, primary, attrs) -> Entity:
    return Entity(ns=ns, noun=noun, plural=plural, primary=primary, attrs=attrs)


# --------------------------------------------------------------------------
# Schemas
#
# Naming conventions are deliberately inconsistent across schemas (kebab-case,
# abbreviated, plain) so the model learns to copy attributes rather than to
# recall a fixed vocabulary.
# --------------------------------------------------------------------------

SCHEMAS = [
    Schema("users", "User management", [
        E("user", "user", "users", "name", [
            A("name", "string", values=['"Alice"', '"John"'], subs=["Dr.", "A"]),
            A("email", "string", unique=True, subs=["gmail", "company.com"]),
            A("age", "long", nums=["18", "30", "65"]),
            A("active", "boolean"),
            A("created", "instant"),
            A("role", "keyword", values=[":admin", ":moderator", ":guest"]),
            A("friends", "ref", many=True, ref_to="user"),
        ]),
    ]),
    Schema("ecommerce", "E-commerce platform", [
        E("product", "product", "products", "name", [
            A("name", "string", subs=["laptop", "Pro"]),
            A("price", "double", nums=["50.0", "100.0", "500.0"]),
            A("category", "keyword", values=[":electronics", ":clothing", ":books"]),
            A("stock", "long", nums=["5", "10", "100"]),
            A("active", "boolean"),
        ]),
        E("order", "order", "orders", "total", [
            A("user", "ref", ref_to="customer"),
            A("products", "ref", many=True, ref_to="product"),
            A("total", "double", nums=["100.0", "500.0", "1000.0"]),
            A("date", "instant"),
            A("status", "keyword", values=[":pending", ":shipped", ":completed"]),
        ]),
        E("customer", "customer", "customers", "name", [
            A("name", "string"),
            A("email", "string", unique=True, subs=["gmail"]),
        ]),
    ]),
    Schema("blog", "Blog platform", [
        E("post", "post", "posts", "title", [
            A("title", "string", fulltext=True, subs=["Clojure"]),
            A("body", "string", fulltext=True),
            A("author", "ref", ref_to="author"),
            A("published", "instant"),
            A("tags", "keyword", many=True, values=[":clojure", ":java", ":programming"]),
            A("views", "long", nums=["100", "1000", "5000"]),
        ]),
        E("comment", "comment", "comments", "text", [
            A("post", "ref", ref_to="post"),
            A("author", "ref", ref_to="author"),
            A("text", "string"),
            A("created", "instant"),
        ]),
        E("author", "author", "authors", "name", [
            A("name", "string"),
            A("bio", "string", fulltext=True),
        ]),
    ]),
    Schema("hr", "HR management", [
        E("employee", "employee", "employees", "name", [
            A("name", "string"),
            A("email", "string", subs=["corp.com"]),
            A("department", "ref", ref_to="department"),
            A("manager", "ref", ref_to="employee"),
            A("salary", "double", nums=["50000.0", "100000.0"]),
            A("hired", "instant"),
            A("title", "string", subs=["Senior"]),
        ]),
        E("department", "department", "departments", "name", [
            A("name", "string"),
            A("budget", "double", nums=["500000.0", "1000000.0"]),
        ]),
    ]),
    Schema("library", "Library system", [
        E("book", "book", "books", "title", [
            A("title", "string", fulltext=True, subs=["adventure"]),
            A("author", "string", subs=["Stephen"]),
            A("isbn", "string", unique=True),
            A("published", "instant"),
            A("genre", "keyword", values=[":fiction", ":mystery", ":science-fiction"]),
            A("pages", "long", nums=["200", "500"]),
        ]),
        E("loan", "loan", "loans", "borrowed", [
            A("book", "ref", ref_to="book"),
            A("member", "ref", ref_to="member"),
            A("borrowed", "instant"),
            A("due", "instant"),
            A("returned", "instant"),
        ]),
        E("member", "member", "members", "name", [
            A("name", "string"),
            A("email", "string"),
        ]),
    ]),
    Schema("inventory", "Inventory tracking", [
        E("item", "item", "items", "name", [
            A("sku", "string", unique=True),
            A("name", "string"),
            A("quantity", "long", nums=["10", "100"]),
            A("location", "ref", ref_to="location"),
            A("reorder-point", "long", nums=["20"]),
            A("unit-cost", "double", nums=["25.0", "50.0"]),
        ]),
        E("location", "location", "locations", "name", [
            A("name", "string"),
            A("type", "keyword", values=[":warehouse", ":retail", ":transit"]),
        ]),
    ]),
    Schema("school", "School records", [
        E("student", "student", "students", "name", [
            A("name", "string"),
            A("grade", "long", nums=["9", "12"]),
            A("gpa", "double", nums=["3.0", "3.5"]),
            A("enrolled", "instant"),
            A("advisor", "ref", ref_to="teacher"),
        ]),
        E("teacher", "teacher", "teachers", "name", [
            A("name", "string"),
            A("subject", "keyword", values=[":math", ":science", ":history"]),
            A("tenured", "boolean"),
        ]),
        E("course", "course", "courses", "title", [
            A("title", "string"),
            A("teacher", "ref", ref_to="teacher"),
            A("credits", "long", nums=["3", "4"]),
            A("capacity", "long", nums=["30"]),
        ]),
    ]),
    Schema("music", "Music catalog", [
        E("track", "track", "tracks", "title", [
            A("title", "string", fulltext=True),
            A("artist", "ref", ref_to="artist"),
            A("album", "ref", ref_to="album"),
            A("duration", "long", nums=["180", "300"]),
            A("plays", "long", nums=["1000", "50000"]),
            A("genre", "keyword", values=[":rock", ":jazz", ":electronic"]),
        ]),
        E("artist", "artist", "artists", "name", [
            A("name", "string"),
            A("country", "keyword", values=[":us", ":uk", ":ng"]),
        ]),
        E("album", "album", "albums", "title", [
            A("title", "string"),
            A("released", "instant"),
        ]),
    ]),
    Schema("restaurant", "Restaurant operations", [
        E("dish", "dish", "dishes", "name", [
            A("name", "string", subs=["chicken"]),
            A("price", "double", nums=["12.0", "25.0"]),
            A("course", "keyword", values=[":appetizer", ":main", ":dessert"]),
            A("vegetarian", "boolean"),
            A("calories", "long", nums=["500", "800"]),
        ]),
        E("reservation", "reservation", "reservations", "time", [
            A("guest", "ref", ref_to="guest"),
            A("time", "instant"),
            A("party-size", "long", nums=["2", "6"]),
            A("confirmed", "boolean"),
        ]),
        E("guest", "guest", "guests", "name", [
            A("name", "string"),
            A("phone", "string", unique=True),
        ]),
    ]),
    Schema("banking", "Banking / accounts", [
        E("acct", "account", "accounts", "num", [
            A("num", "string", unique=True),
            A("bal", "double", nums=["1000.0", "10000.0"], label="balance"),
            A("kind", "keyword", values=[":checking", ":savings", ":credit"]),
            A("holder", "ref", ref_to="cust"),
            A("opened", "instant"),
            A("frozen", "boolean"),
        ]),
        E("cust", "customer", "customers", "name", [
            A("name", "string"),
            A("ssn", "string", unique=True),
            A("credit-score", "long", nums=["650", "750"]),
        ]),
        E("txn", "transaction", "transactions", "amt", [
            A("acct", "ref", ref_to="acct"),
            A("amt", "double", nums=["100.0", "5000.0"], label="amount"),
            A("posted", "instant"),
            A("kind", "keyword", values=[":debit", ":credit"]),
        ]),
    ]),
    Schema("fleet", "Vehicle fleet", [
        E("vehicle", "vehicle", "vehicles", "vin", [
            A("vin", "string", unique=True),
            A("make", "string"),
            A("model", "string"),
            A("mileage", "long", nums=["50000", "150000"]),
            A("in-service", "boolean"),
            A("acquired", "instant"),
            A("driver", "ref", ref_to="driver"),
        ]),
        E("driver", "driver", "drivers", "name", [
            A("name", "string"),
            A("license", "string", unique=True),
            A("status", "keyword", values=[":active", ":suspended"]),
        ]),
    ]),
    Schema("realestate", "Real estate listings", [
        E("listing", "listing", "listings", "address", [
            A("address", "string", fulltext=True),
            A("price", "double", nums=["250000.0", "750000.0"]),
            A("bedrooms", "long", nums=["2", "4"]),
            A("sqft", "long", nums=["1200", "3000"]),
            A("listed", "instant"),
            A("status", "keyword", values=[":available", ":pending", ":sold"]),
            A("agent", "ref", ref_to="agent"),
        ]),
        E("agent", "agent", "agents", "name", [
            A("name", "string"),
            A("brokerage", "string"),
            A("commission", "double", nums=["0.03"]),
        ]),
    ]),
    Schema("issues", "Issue tracker", [
        E("issue", "issue", "issues", "title", [
            A("title", "string", fulltext=True),
            A("body", "string", fulltext=True),
            A("state", "keyword", values=[":open", ":closed"]),
            A("priority", "keyword", values=[":low", ":high", ":urgent"]),
            A("assignee", "ref", ref_to="dev"),
            A("opened", "instant"),
            A("labels", "keyword", many=True, values=[":bug", ":feature"]),
        ]),
        E("dev", "developer", "developers", "handle", [
            A("handle", "string", unique=True),
            A("name", "string"),
            A("commits", "long", nums=["100", "1000"]),
        ]),
    ]),
    Schema("gym", "Gym membership", [
        E("member", "member", "members", "name", [
            A("name", "string"),
            A("joined", "instant"),
            A("tier", "keyword", values=[":basic", ":premium"]),
            A("dues", "double", nums=["29.99", "79.99"]),
            A("active", "boolean"),
        ]),
        E("session", "session", "sessions", "start", [
            A("member", "ref", ref_to="member"),
            A("trainer", "ref", ref_to="trainer"),
            A("start", "instant"),
            A("minutes", "long", nums=["30", "60"]),
        ]),
        E("trainer", "trainer", "trainers", "name", [
            A("name", "string"),
            A("specialty", "keyword", values=[":strength", ":cardio"]),
        ]),
    ]),
    Schema("recipes", "Recipe collection", [
        E("recipe", "recipe", "recipes", "name", [
            A("name", "string", fulltext=True),
            A("instructions", "string", fulltext=True),
            A("prep-minutes", "long", nums=["15", "45"]),
            A("servings", "long", nums=["4"]),
            A("cuisine", "keyword", values=[":italian", ":thai", ":ghanaian"]),
            A("ingredients", "ref", many=True, ref_to="ingredient"),
        ]),
        E("ingredient", "ingredient", "ingredients", "name", [
            A("name", "string"),
            A("cost", "double", nums=["2.5"]),
            A("allergen", "boolean"),
        ]),
    ]),
    Schema("hotel", "Hotel bookings", [
        E("room", "room", "rooms", "number", [
            A("number", "string", unique=True),
            A("rate", "double", nums=["120.0", "350.0"]),
            A("beds", "long", nums=["1", "2"]),
            A("kind", "keyword", values=[":standard", ":suite"]),
            A("smoking", "boolean"),
        ]),
        E("booking", "booking", "bookings", "checkin", [
            A("room", "ref", ref_to="room"),
            A("guest", "ref", ref_to="guest"),
            A("checkin", "instant"),
            A("checkout", "instant"),
            A("nights", "long", nums=["2", "7"]),
        ]),
        E("guest", "guest", "guests", "name", [
            A("name", "string"),
            A("loyalty-id", "string", unique=True),
        ]),
    ]),
    Schema("insurance", "Insurance policies", [
        E("policy", "policy", "policies", "number", [
            A("number", "string", unique=True),
            A("premium", "double", nums=["500.0", "2000.0"]),
            A("kind", "keyword", values=[":auto", ":home", ":life"]),
            A("holder", "ref", ref_to="holder"),
            A("effective", "instant"),
            A("lapsed", "boolean"),
        ]),
        E("claim", "claim", "claims", "amount", [
            A("policy", "ref", ref_to="policy"),
            A("amount", "double", nums=["1000.0", "25000.0"]),
            A("filed", "instant"),
            A("status", "keyword", values=[":filed", ":approved", ":denied"]),
        ]),
        E("holder", "policyholder", "policyholders", "name", [
            A("name", "string"),
            A("dob", "instant"),
        ]),
    ]),
    Schema("farm", "Farm management", [
        E("field", "field", "fields", "name", [
            A("name", "string"),
            A("acres", "double", nums=["40.0", "200.0"]),
            A("crop", "keyword", values=[":maize", ":cassava", ":soy"]),
            A("irrigated", "boolean"),
        ]),
        E("harvest", "harvest", "harvests", "date", [
            A("field", "ref", ref_to="field"),
            A("date", "instant"),
            A("yield-tons", "double", nums=["10.0", "50.0"]),
        ]),
    ]),
    Schema("legal", "Legal case management", [
        E("case", "case", "cases", "caption", [
            A("caption", "string", fulltext=True),
            A("filed", "instant"),
            A("court", "keyword", values=[":district", ":appellate"]),
            A("lead", "ref", ref_to="atty"),
            A("closed", "boolean"),
        ]),
        E("atty", "attorney", "attorneys", "name", [
            A("name", "string"),
            A("bar-no", "string", unique=True),
            A("rate", "double", nums=["350.0", "800.0"]),
        ]),
    ]),
    Schema("events", "Event management", [
        E("event", "event", "events", "title", [
            A("title", "string", fulltext=True),
            A("starts", "instant"),
            A("capacity", "long", nums=["50", "500"]),
            A("venue", "ref", ref_to="venue"),
            A("category", "keyword", values=[":conference", ":concert", ":workshop"]),
        ]),
        E("ticket", "ticket", "tickets", "price", [
            A("event", "ref", ref_to="event"),
            A("attendee", "ref", ref_to="attendee"),
            A("price", "double", nums=["25.0", "150.0"]),
            A("scanned", "boolean"),
        ]),
        E("attendee", "attendee", "attendees", "name", [
            A("name", "string"),
            A("email", "string", unique=True, subs=["gmail"]),
        ]),
        E("venue", "venue", "venues", "name", [
            A("name", "string"),
            A("city", "string"),
        ]),
    ]),
    Schema("crm", "Sales CRM", [
        E("lead", "lead", "leads", "company", [
            A("company", "string"),
            A("value", "double", nums=["5000.0", "50000.0"]),
            A("stage", "keyword", values=[":new", ":qualified", ":won", ":lost"]),
            A("owner", "ref", ref_to="rep"),
            A("created", "instant"),
        ]),
        E("rep", "sales rep", "sales reps", "name", [
            A("name", "string"),
            A("quota", "double", nums=["100000.0"]),
            A("region", "keyword", values=[":emea", ":apac", ":amer"]),
        ]),
    ]),
    Schema("telecom", "Telecom subscribers", [
        E("sub", "subscriber", "subscribers", "msisdn", [
            A("msisdn", "string", unique=True),
            A("plan", "ref", ref_to="plan"),
            A("data-mb", "long", nums=["1000", "20000"]),
            A("activated", "instant"),
            A("roaming", "boolean"),
        ]),
        E("plan", "plan", "plans", "name", [
            A("name", "string"),
            A("monthly", "double", nums=["15.0", "60.0"]),
            A("tier", "keyword", values=[":prepaid", ":postpaid"]),
        ]),
    ]),
    Schema("museum", "Museum collection", [
        E("artifact", "artifact", "artifacts", "title", [
            A("title", "string", fulltext=True),
            A("provenance", "string", fulltext=True),
            A("acquired", "instant"),
            A("era", "keyword", values=[":ancient", ":medieval", ":modern"]),
            A("appraisal", "double", nums=["10000.0", "500000.0"]),
            A("on-display", "boolean"),
        ]),
        E("exhibit", "exhibit", "exhibits", "name", [
            A("name", "string"),
            A("curator", "ref", ref_to="curator"),
            A("opens", "instant"),
        ]),
        E("curator", "curator", "curators", "name", [
            A("name", "string"),
            A("tenure-years", "long", nums=["5", "20"]),
        ]),
    ]),
    Schema("lab", "Laboratory samples", [
        E("sample", "sample", "samples", "barcode", [
            A("barcode", "string", unique=True),
            A("collected", "instant"),
            A("kind", "keyword", values=[":blood", ":tissue", ":swab"]),
            A("volume-ml", "double", nums=["2.5", "10.0"]),
            A("contaminated", "boolean"),
            A("assay", "ref", ref_to="assay"),
        ]),
        E("assay", "assay", "assays", "name", [
            A("name", "string"),
            A("turnaround-hours", "long", nums=["24", "72"]),
        ]),
    ]),
    Schema("shipping", "Freight and shipments", [
        E("shipment", "shipment", "shipments", "tracking", [
            A("tracking", "string", unique=True),
            A("weight-kg", "double", nums=["5.0", "500.0"]),
            A("shipped", "instant"),
            A("delivered", "instant"),
            A("status", "keyword", values=[":in-transit", ":delivered", ":lost"]),
            A("carrier", "ref", ref_to="carrier"),
        ]),
        E("carrier", "carrier", "carriers", "name", [
            A("name", "string"),
            A("scac", "string", unique=True),
        ]),
    ]),
    Schema("sports", "Sports league", [
        E("player", "player", "players", "name", [
            A("name", "string"),
            A("position", "keyword", values=[":forward", ":midfield", ":keeper"]),
            A("goals", "long", nums=["5", "20"]),
            A("team", "ref", ref_to="team"),
            A("debut", "instant"),
        ]),
        E("team", "team", "teams", "name", [
            A("name", "string"),
            A("city", "string"),
            A("founded", "instant"),
            A("payroll", "double", nums=["1000000.0"]),
        ]),
    ]),

    # ---- held out for validation ----
    Schema("clinic", "Medical clinic", [
        E("patient", "patient", "patients", "name", [
            A("name", "string"),
            A("mrn", "string", unique=True),
            A("dob", "instant"),
            A("bloodtype", "keyword", values=[":a-pos", ":o-neg"]),
            A("primary-doc", "ref", ref_to="doctor"),
        ]),
        E("visit", "visit", "visits", "date", [
            A("patient", "ref", ref_to="patient"),
            A("doctor", "ref", ref_to="doctor"),
            A("date", "instant"),
            A("charge", "double", nums=["150.0", "900.0"]),
            A("followup", "boolean"),
        ]),
        E("doctor", "doctor", "doctors", "name", [
            A("name", "string"),
            A("specialty", "keyword", values=[":cardiology", ":pediatrics"]),
            A("npi", "string", unique=True),
        ]),
    ]),
    Schema("airline", "Airline operations", [
        E("flight", "flight", "flights", "number", [
            A("number", "string"),
            A("departs", "instant"),
            A("duration-min", "long", nums=["90", "600"]),
            A("aircraft", "ref", ref_to="aircraft"),
            A("status", "keyword", values=[":scheduled", ":delayed", ":cancelled"]),
        ]),
        E("aircraft", "aircraft", "aircraft", "tail", [
            A("tail", "string", unique=True),
            A("model", "string"),
            A("seats", "long", nums=["150", "300"]),
            A("hours", "double", nums=["10000.0"]),
        ]),
        E("pax", "passenger", "passengers", "name", [
            A("name", "string"),
            A("flight", "ref", ref_to="flight"),
            A("fare", "double", nums=["200.0", "1500.0"]),
            A("checked-in", "boolean"),
        ]),
    ]),
    Schema("podcast", "Podcast network", [
        E("episode", "episode", "episodes", "title", [
            A("title", "string", fulltext=True),
            A("notes", "string", fulltext=True),
            A("published", "instant"),
            A("duration-sec", "long", nums=["1800", "5400"]),
            A("downloads", "long", nums=["1000", "100000"]),
            A("show", "ref", ref_to="show"),
        ]),
        E("show", "show", "shows", "name", [
            A("name", "string"),
            A("category", "keyword", values=[":tech", ":comedy", ":news"]),
            A("host", "ref", ref_to="host"),
        ]),
        E("host", "host", "hosts", "name", [
            A("name", "string"),
            A("followers", "long", nums=["5000", "200000"]),
        ]),
    ]),
    Schema("parking", "Parking garage", [
        E("spot", "spot", "spots", "code", [
            A("code", "string", unique=True),
            A("level", "long", nums=["1", "5"]),
            A("kind", "keyword", values=[":compact", ":ev", ":handicap"]),
            A("occupied", "boolean"),
            A("hourly", "double", nums=["3.5", "8.0"]),
        ]),
        E("permit", "permit", "permits", "plate", [
            A("plate", "string", unique=True),
            A("spot", "ref", ref_to="spot"),
            A("issued", "instant"),
            A("expires", "instant"),
        ]),
    ]),
    Schema("bugtracker", "Defect tracking", [
        E("defect", "defect", "defects", "summary", [
            A("summary", "string", fulltext=True),
            A("repro", "string", fulltext=True),
            A("sev", "keyword", values=[":s1", ":s2", ":s3"], label="severity"),
            A("found", "instant"),
            A("fixed", "instant"),
            A("reporter", "ref", ref_to="qa"),
            A("dupe-of", "ref", ref_to="defect"),
        ]),
        E("qa", "tester", "testers", "name", [
            A("name", "string"),
            A("team", "keyword", values=[":core", ":platform"]),
        ]),
    ]),
]


# --------------------------------------------------------------------------
# Pattern generators
#
# Each takes a Schema and returns Examples. They walk the typed schema, so a
# new schema automatically gets full coverage.
# --------------------------------------------------------------------------

YEARS = ["2020", "2022", "2024"]

# Attribute names whose literal spelling reads badly in a prompt. Keeps the
# natural-language side sounding like something a person would actually type.
DEFAULT_LABELS = {
    "duration-min": "duration", "duration-sec": "duration", "data-mb": "data usage",
    "yield-tons": "yield", "volume-ml": "volume", "weight-kg": "weight",
    "prep-minutes": "prep time", "tenure-years": "tenure",
    "turnaround-hours": "turnaround", "bar-no": "bar number",
    "msisdn": "phone number", "npi": "NPI", "scac": "SCAC code",
    "mrn": "medical record number", "vin": "VIN", "ssn": "SSN", "gpa": "GPA",
    "dob": "date of birth", "sqft": "square footage", "checkin": "check-in date",
    "checkout": "check-out date", "dupe-of": "duplicate of",
    "in-service": "in service", "on-display": "on display",
    "checked-in": "checked in", "primary-doc": "primary doctor",
    "monthly": "monthly rate", "hourly": "hourly rate", "hours": "airframe hours",
    "departs": "departure time", "starts": "start time", "opens": "opening date",
    "expires": "expiry date", "due": "due date", "effective": "effective date",
    "found": "date found", "fixed": "date fixed", "num": "number",
    "amt": "amount", "bal": "balance", "sev": "severity", "kind": "type",
}

# Substrings for string predicates, so more string attrs get realistic coverage
# without annotating every schema by hand.
DEFAULT_SUBS = {
    "name": ["Smith"], "title": ["Guide"], "email": ["gmail"],
    "address": ["Main"], "city": ["Accra"], "model": ["X"],
    "make": ["Toyota"], "company": ["Corp"], "caption": ["Estate"],
    "summary": ["crash"], "handle": ["dev"], "plate": ["GR"],
    "tracking": ["1Z"], "number": ["A"], "code": ["B"],
}


def g_basic(s: Schema) -> List[Example]:
    out = []
    for e in s.entities:
        find_all = f"[:find ?e :where [?e {e.primary_kw} _]]"
        for nl in (f"Find all {e.plural}",
                   f"Get all {e.plural}",
                   f"Show me all {e.plural}",
                   f"List every {e.noun}"):
            out.append(Example(s, nl, find_all, "basic"))

        scalars = [a for a in e.scalars() if not a.many]
        for a in scalars[:3]:
            out.append(Example(
                s, f"Find all {e.noun} {a.words()}s",
                f"[:find ?v :where [_ {e.kw(a.name)} ?v]]", "basic"))
        if len(scalars) >= 2:
            a, b = scalars[0], scalars[1]
            out.append(Example(
                s, f"Get {e.noun} {a.words()} and {b.words()}",
                f"[:find ?a ?b :where [?e {e.kw(a.name)} ?a] [?e {e.kw(b.name)} ?b]]",
                "basic"))
        if len(scalars) >= 3:
            a, b, c = scalars[0], scalars[1], scalars[2]
            out.append(Example(
                s, f"Show {e.noun} {a.words()}, {b.words()} and {c.words()}",
                f"[:find ?a ?b ?c :where [?e {e.kw(a.name)} ?a] "
                f"[?e {e.kw(b.name)} ?b] [?e {e.kw(c.name)} ?c]]", "basic"))
    return out


def g_equality(s: Schema) -> List[Example]:
    out = []
    for e in s.entities:
        for a in e.attrs:
            kw = e.kw(a.name)
            if a.type == "boolean":
                out.append(Example(s, f"Find {a.words()} {e.plural}",
                                   f"[:find ?e :where [{'?e'} {kw} true]]", "filter"))
                out.append(Example(s, f"Find {e.plural} where {a.words()} is false",
                                   f"[:find ?e :where [?e {kw} false]]", "filter"))
            for v in a.values:
                pretty = v.lstrip(':').strip('"')
                if a.type == "keyword":
                    out.append(Example(s, f"Find {e.plural} with {a.words()} {pretty}",
                                       f"[:find ?e :where [?e {kw} {v}]]", "filter"))
                elif a.type == "string":
                    out.append(Example(s, f"Find {e.plural} whose {a.words()} is {pretty}",
                                       f"[:find ?e :where [?e {kw} {v}]]", "filter"))
    return out


def g_comparison(s: Schema) -> List[Example]:
    out = []
    for e in s.entities:
        for a in e.by_type("long", "double"):
            kw, w = e.kw(a.name), a.words()
            for n in a.nums:
                out.append(Example(s, f"Find {e.plural} with {w} greater than {n}",
                                   f"[:find ?e :where [?e {kw} ?v] [(> ?v {n})]]", "comparison"))
                out.append(Example(s, f"Find {e.plural} with {w} under {n}",
                                   f"[:find ?e :where [?e {kw} ?v] [(< ?v {n})]]", "comparison"))
                out.append(Example(s, f"Find {e.plural} with {w} of at least {n}",
                                   f"[:find ?e :where [?e {kw} ?v] [(>= ?v {n})]]", "comparison"))
            if len(a.nums) >= 2:
                lo, hi = a.nums[0], a.nums[1]
                out.append(Example(s, f"Find {e.plural} with {w} between {lo} and {hi}",
                                   f"[:find ?e :where [?e {kw} ?v] [(>= ?v {lo})] [(<= ?v {hi})]]",
                                   "comparison"))
        nums = e.by_type("long", "double")
        if len(nums) >= 2:
            a, b = nums[0], nums[1]
            out.append(Example(
                s, f"Find {e.plural} where {a.words()} is below {b.words()}",
                f"[:find ?e :where [?e {e.kw(a.name)} ?x] [?e {e.kw(b.name)} ?y] [(< ?x ?y)]]",
                "comparison"))
    return out


def g_string(s: Schema) -> List[Example]:
    out = []
    for e in s.entities:
        for a in e.by_type("string"):
            kw, w = e.kw(a.name), a.words()
            for sub in a.substrings():
                out.append(Example(
                    s, f"Find {e.plural} whose {w} contains {sub}",
                    f'[:find ?e :where [?e {kw} ?v] [(clojure.string/includes? ?v "{sub}")]]',
                    "string"))
                out.append(Example(
                    s, f"Find {e.plural} whose {w} starts with {sub}",
                    f'[:find ?e :where [?e {kw} ?v] [(clojure.string/starts-with? ?v "{sub}")]]',
                    "string"))
                out.append(Example(
                    s, f"Find {e.plural} whose {w} ends with {sub}",
                    f'[:find ?e :where [?e {kw} ?v] [(clojure.string/ends-with? ?v "{sub}")]]',
                    "string"))
    return out


def g_dates(s: Schema) -> List[Example]:
    out = []
    for e in s.entities:
        for a in e.by_type("instant"):
            kw, w = e.kw(a.name), a.words()

            def phrase(rel: str, tail: str) -> str:
                # Participle attrs read directly ("posts published after 2024");
                # noun attrs need a frame ("permits with expiry date after 2024").
                if a.participle:
                    return f"Find {e.plural} {a.name.replace('-', ' ')} {rel} {tail}"
                return f"Find {e.plural} with {w} {rel} {tail}"

            for y in YEARS:
                out.append(Example(
                    s, phrase("after", y),
                    f'[:find ?e :where [?e {kw} ?d] [(> ?d #inst "{y}-01-01")]]', "date"))
                out.append(Example(
                    s, phrase("before", y),
                    f'[:find ?e :where [?e {kw} ?d] [(< ?d #inst "{y}-01-01")]]', "date"))
                out.append(Example(
                    s, phrase("during", y),
                    f'[:find ?e :where [?e {kw} ?d] [(>= ?d #inst "{y}-01-01")] '
                    f'[(< ?d #inst "{int(y)+1}-01-01")]]', "date"))
            out.append(Example(
                s, phrase("before", "a given date"),
                f"[:find ?e :in $ ?cutoff :where [?e {kw} ?d] [(< ?d ?cutoff)]]", "date"))
    return out


def g_aggregations(s: Schema) -> List[Example]:
    out = []
    for e in s.entities:
        cnt = f"[:find (count ?e) :where [?e {e.primary_kw} _]]"
        out.append(Example(s, f"Count all {e.plural}", cnt, "aggregation"))
        out.append(Example(s, f"How many {e.plural} are there", cnt, "aggregation"))

        for a in e.by_type("long", "double"):
            kw, w = e.kw(a.name), a.words()
            for fn, phrase in (("sum", f"Total {w} across all {e.plural}"),
                               ("avg", f"Average {e.noun} {w}"),
                               ("max", f"Highest {e.noun} {w}"),
                               ("min", f"Lowest {e.noun} {w}"),
                               ("median", f"Median {e.noun} {w}")):
                out.append(Example(s, phrase,
                                   f"[:find ({fn} ?v) :where [?e {kw} ?v]]", "aggregation"))

        for a in e.by_type("keyword"):
            out.append(Example(
                s, f"Count {e.plural} by {a.words()}",
                f"[:find ?g (count ?e) :where [?e {e.kw(a.name)} ?g]]", "aggregation"))
    return out


def g_joins(s: Schema) -> List[Example]:
    out = []
    for e in s.entities:
        for a in e.attrs:
            if a.type != "ref" or not a.ref_to:
                continue
            t = s.entity(a.ref_to)
            if t is None:
                continue
            ekw, akw, tkw = e.primary_kw, e.kw(a.name), t.primary_kw
            out.append(Example(
                s, f"Find {e.plural} with their {a.words()} {t.primary}",
                f"[:find ?x ?y :where [?e {ekw} ?x] [?e {akw} ?t] [?t {tkw} ?y]]", "join"))
            out.append(Example(
                s, f"Show each {e.noun} and the {t.noun} it links to",
                f"[:find ?x ?y :where [?e {ekw} ?x] [?e {akw} ?t] [?t {tkw} ?y]]", "join"))
            for k in t.by_type("keyword")[:1]:
                for v in k.values[:1]:
                    out.append(Example(
                        s, f"Find {e.plural} whose {t.noun} has {k.words()} {v.lstrip(':')}",
                        f"[:find ?x :where [?e {ekw} ?x] [?e {akw} ?t] [?t {t.kw(k.name)} {v}]]",
                        "join"))
    return out


def g_negations(s: Schema) -> List[Example]:
    out = []
    for e in s.entities:
        for a in e.attrs:
            if a.name == e.primary:
                continue
            out.append(Example(
                s, f"Find {e.plural} with no {a.words()}",
                f"[:find ?e :where [?e {e.primary_kw} _] (not [?e {e.kw(a.name)} _])]",
                "negation"))
        for a in e.by_type("keyword"):
            for v in a.values[:1]:
                out.append(Example(
                    s, f"Find {e.plural} whose {a.words()} is not {v.lstrip(':')}",
                    f"[:find ?e :where [?e {e.primary_kw} _] (not [?e {e.kw(a.name)} {v}])]",
                    "negation"))
    # reverse-reference negation: entities nothing points at
    for e in s.entities:
        for a in e.attrs:
            if a.type == "ref" and a.ref_to and s.entity(a.ref_to):
                t = s.entity(a.ref_to)
                out.append(Example(
                    s, f"Find {t.plural} not referenced by any {e.noun}",
                    f"[:find ?t :where [?t {t.primary_kw} _] (not [?e {e.kw(a.name)} ?t])]",
                    "negation"))
    return out


def g_pull(s: Schema) -> List[Example]:
    out = []
    for e in s.entities:
        out.append(Example(
            s, f"Get full {e.noun} details",
            f"[:find (pull ?e [*]) :where [?e {e.primary_kw} _]]", "pull"))
        scalars = [a for a in e.scalars() if not a.many][:2]
        if len(scalars) == 2:
            out.append(Example(
                s, f"Get only {e.noun} {scalars[0].words()} and {scalars[1].words()}",
                f"[:find (pull ?e [{e.kw(scalars[0].name)} {e.kw(scalars[1].name)}]) "
                f":where [?e {e.primary_kw} _]]", "pull"))
        for a in e.attrs:
            if a.type == "ref" and a.ref_to and s.entity(a.ref_to):
                t = s.entity(a.ref_to)
                out.append(Example(
                    s, f"Get {e.plural} with their {a.words()} expanded",
                    f"[:find (pull ?e [{e.primary_kw} {{{e.kw(a.name)} [{t.primary_kw}]}}]) "
                    f":where [?e {e.primary_kw} _]]", "pull"))
                break
    return out


def g_or(s: Schema) -> List[Example]:
    out = []
    for e in s.entities:
        for a in e.by_type("keyword"):
            if len(a.values) < 2:
                continue
            v1, v2 = a.values[0], a.values[1]
            out.append(Example(
                s, f"Find {e.plural} with {a.words()} {v1.lstrip(':')} or {v2.lstrip(':')}",
                f"[:find ?e :where (or [?e {e.kw(a.name)} {v1}] [?e {e.kw(a.name)} {v2}])]",
                "or"))
        nums = e.by_type("long", "double")
        if nums and len(nums[0].nums) >= 2:
            a = nums[0]
            lo, hi = a.nums[0], a.nums[1]
            out.append(Example(
                s, f"Find {e.plural} with {a.words()} below {lo} or above {hi}",
                f"[:find ?e :where [?e {e.kw(a.name)} ?v] (or [(< ?v {lo})] [(> ?v {hi})])]",
                "or"))
    return out


def g_return_formats(s: Schema) -> List[Example]:
    out = []
    for e in s.entities:
        for a in e.by_type("keyword", "string")[:2]:
            out.append(Example(
                s, f"Get all distinct {e.noun} {a.words()} values as a list",
                f"[:find [?v ...] :where [_ {e.kw(a.name)} ?v]]", "return-format"))
        uniq = [a for a in e.attrs if a.unique]
        if uniq:
            a = uniq[0]
            out.append(Example(
                s, f"Get the single {e.noun} with a given {a.words()}",
                f"[:find ?e . :in $ ?v :where [?e {e.kw(a.name)} ?v]]", "return-format"))
        scalars = [a for a in e.scalars() if not a.many][:2]
        if len(scalars) == 2:
            n1, n2 = scalars[0].name.replace("-", "_"), scalars[1].name.replace("-", "_")
            out.append(Example(
                s, f"Get {e.plural} as maps keyed by {scalars[0].words()} and {scalars[1].words()}",
                f"[:find ?a ?b :keys {n1} {n2} :where "
                f"[?e {e.kw(scalars[0].name)} ?a] [?e {e.kw(scalars[1].name)} ?b]]",
                "return-format"))
    return out


def g_inputs(s: Schema) -> List[Example]:
    out = []
    for e in s.entities:
        for a in e.by_type("keyword", "string")[:2]:
            out.append(Example(
                s, f"Find {e.plural} by a given {a.words()}",
                f"[:find ?e :in $ ?v :where [?e {e.kw(a.name)} ?v]]", "input"))
            out.append(Example(
                s, f"Find {e.plural} matching any of several {a.words()} values",
                f"[:find ?e :in $ [?v ...] :where [?e {e.kw(a.name)} ?v]]", "input"))
        for a in e.by_type("long", "double")[:1]:
            out.append(Example(
                s, f"Find {e.plural} with {a.words()} in a given range",
                f"[:find ?e :in $ ?min ?max :where [?e {e.kw(a.name)} ?v] "
                f"[(>= ?v ?min)] [(<= ?v ?max)]]", "input"))
    return out


def g_fulltext(s: Schema) -> List[Example]:
    """Datalevin-specific full-text search."""
    out = []
    for e in s.entities:
        for a in e.attrs:
            if not a.fulltext:
                continue
            kw = e.kw(a.name)
            term = a.subs[0] if a.subs else e.noun
            out.append(Example(
                s, f"Search {e.plural} for {term}",
                f'[:find ?e ?a ?v :where [(fulltext $ "{term}") [[?e ?a ?v]]]]', "fulltext"))
            out.append(Example(
                s, f"Search {e.noun} {a.words()} for {term}",
                f'[:find ?e ?a ?v :where [(fulltext $ {kw} "{term}") [[?e ?a ?v]]]]', "fulltext"))
            out.append(Example(
                s, f"Search {e.plural} for the exact phrase {term} guide",
                f'[:find ?e ?a ?v :where [(fulltext $ {{:phrase "{term} guide"}}) '
                f'[[?e ?a ?v]]]]', "fulltext"))
            out.append(Example(
                s, f"Search {e.plural} for {term} but not draft",
                f'[:find ?e ?a ?v :where [(fulltext $ [:and "{term}" [:not "draft"]]) '
                f'[[?e ?a ?v]]]]', "fulltext"))
            out.append(Example(
                s, f"Search {e.plural} for {term} or archive",
                f'[:find ?e ?a ?v :where [(fulltext $ [:or "{term}" "archive"]) '
                f'[[?e ?a ?v]]]]', "fulltext"))
    return out


def g_complex(s: Schema) -> List[Example]:
    """Multi-clause compositions -- the shapes users actually struggle to write."""
    out = []
    for e in s.entities:
        nums = e.by_type("long", "double")
        refs = [a for a in e.attrs if a.type == "ref" and a.ref_to and s.entity(a.ref_to)]
        kws = e.by_type("keyword")
        bools = e.by_type("boolean")

        # numeric filter + join projection
        if nums and refs and nums[0].nums:
            a, r = nums[0], refs[0]
            t = s.entity(r.ref_to)
            out.append(Example(
                s, f"Find {e.plural} with {a.words()} over {a.nums[0]} "
                   f"along with their {t.noun} {t.primary}",
                f"[:find ?x ?y :where [?e {e.primary_kw} ?x] [?e {e.kw(a.name)} ?v] "
                f"[(> ?v {a.nums[0]})] [?e {e.kw(r.name)} ?t] [?t {t.primary_kw} ?y]]",
                "complex"))

        # group-by across a join
        if nums and refs and nums[0].nums:
            a, r = nums[0], refs[0]
            t = s.entity(r.ref_to)
            out.append(Example(
                s, f"Average {e.noun} {a.words()} grouped by {t.noun}",
                f"[:find ?y (avg ?v) :where [?e {e.kw(a.name)} ?v] "
                f"[?e {e.kw(r.name)} ?t] [?t {t.primary_kw} ?y]]", "complex"))
            out.append(Example(
                s, f"Count {e.plural} per {t.noun}",
                f"[:find ?y (count ?e) :where [?e {e.kw(r.name)} ?t] [?t {t.primary_kw} ?y]]",
                "complex"))

        # boolean + keyword + numeric stacked
        if bools and kws and nums and kws[0].values and nums[0].nums:
            b, k, n = bools[0], kws[0], nums[0]
            out.append(Example(
                s, f"Find {b.words()} {e.plural} with {k.words()} "
                   f"{k.values[0].lstrip(':')} and {n.words()} above {n.nums[0]}",
                f"[:find ?e :where [?e {e.kw(b.name)} true] [?e {e.kw(k.name)} {k.values[0]}] "
                f"[?e {e.kw(n.name)} ?v] [(> ?v {n.nums[0]})]]", "complex"))

        # date filter + negation + join (the "overdue loans" shape)
        dates = e.by_type("instant")
        if len(dates) >= 2 and refs:
            d1, d2 = dates[0], dates[1]
            r = refs[0]
            t = s.entity(r.ref_to)
            out.append(Example(
                s, f"Find {e.plural} past {d2.words()} with no {d1.words()}, "
                   f"including {t.noun} {t.primary}",
                f"[:find ?y ?d :in $ ?now :where [?e {e.kw(d2.name)} ?d] [(< ?d ?now)] "
                f"(not [?e {e.kw(d1.name)} _]) [?e {e.kw(r.name)} ?t] [?t {t.primary_kw} ?y]]",
                "complex"))

        # computed binding
        if len(nums) >= 2:
            a, b = nums[0], nums[1]
            out.append(Example(
                s, f"Total {a.words()} times {b.words()} across {e.plural}",
                f"[:find (sum ?p) :where [?e {e.kw(a.name)} ?x] [?e {e.kw(b.name)} ?y] "
                f"[(* ?x ?y) ?p]]", "complex"))
    return out


GENERATORS = [
    g_basic, g_equality, g_comparison, g_string, g_dates, g_aggregations,
    g_joins, g_negations, g_pull, g_or, g_return_formats, g_inputs,
    g_fulltext, g_complex,
]


# --------------------------------------------------------------------------
# Validation -- catch generator bugs before they become training targets
# --------------------------------------------------------------------------

ATTR_RE = re.compile(r':[a-z][\w-]*/[\w-]+')
PAIRS = {')': '(', ']': '[', '}': '{'}


def balanced(q: str) -> bool:
    stack = []
    in_str = False
    i = 0
    while i < len(q):
        c = q[i]
        if in_str:
            if c == '\\':
                i += 2
                continue
            if c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c in '([{':
            stack.append(c)
        elif c in ')]}':
            if not stack or stack.pop() != PAIRS[c]:
                return False
        i += 1
    return not stack and not in_str


def validate(ex: Example) -> List[str]:
    errs = []
    if not balanced(ex.query):
        errs.append("unbalanced delimiters")
    if not ex.query.startswith("[:find"):
        errs.append("does not start with [:find")
    known = ex.schema.attr_kws
    for kw in ATTR_RE.findall(ex.query):
        if kw.startswith(":db"):
            continue
        if kw not in known:
            errs.append(f"attribute {kw} not in schema")
    # every ?var in :find must be bound somewhere in :where
    head, _, tail = ex.query.partition(":where")
    for v in set(re.findall(r'\?[\w-]+', head)):
        if v not in tail and v not in head.split(":in")[-1]:
            errs.append(f"{v} in :find is unbound")
    return errs


# --------------------------------------------------------------------------

# Prompt format must match the base model's chat template exactly. Training with
# the wrong control tokens is silent -- they tokenize as ordinary text and the
# model never learns the turn boundaries. Qwen uses ChatML; SmolLM does not.
CHAT_TEMPLATES = {
    "chatml": (
        "<|im_start|>system\n{system}<|im_end|>\n"
        "<|im_start|>user\nSchema: {schema}\n\n{nl}<|im_end|>\n"
        "<|im_start|>assistant\n{query}<|im_end|>"
    ),
    "smollm": "<|user|>Schema: {schema}\n\n{nl}<|assistant|>{query}",
}

SYSTEM = ("You translate natural language into Datalevin Datalog queries. "
          "Use only attributes present in the given schema. Output only the query.")

TEMPLATE = "chatml"   # base model is Qwen2.5-Coder


def format_example(ex: Example) -> str:
    text = CHAT_TEMPLATES[TEMPLATE].format(
        system=SYSTEM, schema=ex.schema.edn, nl=ex.natural, query=ex.query)
    return json.dumps({"text": text})


def main():
    print("Generating Datalevin NLQ training data...\n")

    examples, problems = [], []
    for schema in SCHEMAS:
        for gen in GENERATORS:
            for ex in gen(schema):
                errs = validate(ex)
                if errs:
                    problems.append((schema.name, ex.natural, ex.query, errs))
                else:
                    examples.append(ex)

    if problems:
        print(f"!! {len(problems)} invalid examples dropped:")
        for name, nl, q, errs in problems[:10]:
            print(f"   [{name}] {nl}\n     {q}\n     -> {'; '.join(errs)}")
        print()

    # Dedupe on (schema, natural language). Identical prompts with divergent
    # targets are worse than useless -- they teach the model the task is random.
    seen, deduped = set(), []
    collisions = 0
    for ex in examples:
        key = (ex.schema.name, ex.natural)
        if key in seen:
            collisions += 1
            continue
        seen.add(key)
        deduped.append(ex)
    examples = deduped

    # Rebalance. Template expansion massively over-produces the easy patterns
    # (every numeric attr x every threshold), while joins and multi-clause
    # compositions are limited by schema structure. Left alone the mix is the
    # inverse of what is worth learning, so cap the cheap categories per schema
    # and let the structurally-scarce ones through untouched.
    caps = {"basic": 10, "aggregation": 16, "comparison": 14, "date": 10,
            "negation": 10, "input": 8, "return-format": 7, "pull": 7, "filter": 8}
    rng = random.Random(SEED)
    buckets = {}
    for ex in examples:
        buckets.setdefault((ex.schema.name, ex.category), []).append(ex)
    capped = []
    dropped = 0
    for (_, cat), group in buckets.items():
        limit = caps.get(cat)
        if limit is not None and len(group) > limit:
            dropped += len(group) - limit
            group = rng.sample(group, limit)
        capped.extend(group)
    examples = capped

    train = [e for e in examples if e.schema.name not in HOLDOUT]
    valid = [e for e in examples if e.schema.name in HOLDOUT]

    rng.shuffle(train)
    rng.shuffle(valid)

    cats = {}
    for ex in examples:
        cats[ex.category] = cats.get(ex.category, 0) + 1
    print("Examples by category:")
    for cat, n in sorted(cats.items(), key=lambda kv: -kv[1]):
        print(f"  {cat:<15} {n}")

    print(f"\nSchemas:  {len(SCHEMAS)} total "
          f"({len(SCHEMAS) - len(HOLDOUT)} train / {len(HOLDOUT)} held out)")
    print(f"Held out: {', '.join(sorted(HOLDOUT))}")
    print(f"Dropped {collisions} duplicate prompts, {dropped} over-represented")
    print(f"\nTrain: {len(train)} examples")
    print(f"Valid: {len(valid)} examples  (schema-disjoint from train)")

    with open(TRAIN_FILE, 'w') as f:
        for ex in train:
            f.write(format_example(ex) + '\n')
    with open(VALID_FILE, 'w') as f:
        for ex in valid:
            f.write(format_example(ex) + '\n')

    print(f"\nWrote:\n  {TRAIN_FILE}\n  {VALID_FILE}")


if __name__ == "__main__":
    main()
