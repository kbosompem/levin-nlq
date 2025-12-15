#!/usr/bin/env python3
"""
Generate training data for Datalevin NLQ model.

Produces JSONL files in the format expected by MLX fine-tuning:
{"text": "<|user|>Schema: {...}\n\nQuery<|assistant|>[:find ...]"}
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass

# Output paths
OUTPUT_DIR = Path(__file__).parent.parent / "training-data"
OUTPUT_DIR.mkdir(exist_ok=True)

TRAIN_FILE = OUTPUT_DIR / "train.jsonl"
VALID_FILE = OUTPUT_DIR / "valid.jsonl"

# Validation split ratio
VALID_RATIO = 0.1


@dataclass
class Schema:
    """A sample schema for training examples."""
    name: str
    edn: str
    description: str


@dataclass
class Example:
    """A single training example."""
    schema: Schema
    natural: str
    query: str
    category: str


# Sample schemas covering different domains
SCHEMAS = [
    Schema(
        name="users",
        edn="""{:user/name {:db/valueType :db.type/string}
 :user/email {:db/valueType :db.type/string :db/unique :db.unique/identity}
 :user/age {:db/valueType :db.type/long}
 :user/active {:db/valueType :db.type/boolean}
 :user/created {:db/valueType :db.type/instant}
 :user/role {:db/valueType :db.type/keyword}
 :user/friends {:db/valueType :db.type/ref :db/cardinality :db.cardinality/many}}""",
        description="User management system"
    ),
    Schema(
        name="ecommerce",
        edn="""{:product/name {:db/valueType :db.type/string}
 :product/price {:db/valueType :db.type/double}
 :product/category {:db/valueType :db.type/keyword}
 :product/stock {:db/valueType :db.type/long}
 :product/active {:db/valueType :db.type/boolean}
 :order/user {:db/valueType :db.type/ref}
 :order/products {:db/valueType :db.type/ref :db/cardinality :db.cardinality/many}
 :order/total {:db/valueType :db.type/double}
 :order/date {:db/valueType :db.type/instant}
 :order/status {:db/valueType :db.type/keyword}}""",
        description="E-commerce platform"
    ),
    Schema(
        name="blog",
        edn="""{:post/title {:db/valueType :db.type/string :db/fulltext true}
 :post/body {:db/valueType :db.type/string :db/fulltext true}
 :post/author {:db/valueType :db.type/ref}
 :post/published {:db/valueType :db.type/instant}
 :post/tags {:db/valueType :db.type/keyword :db/cardinality :db.cardinality/many}
 :post/views {:db/valueType :db.type/long}
 :comment/post {:db/valueType :db.type/ref}
 :comment/author {:db/valueType :db.type/ref}
 :comment/text {:db/valueType :db.type/string}
 :comment/created {:db/valueType :db.type/instant}
 :author/name {:db/valueType :db.type/string}
 :author/bio {:db/valueType :db.type/string :db/fulltext true}}""",
        description="Blog platform"
    ),
    Schema(
        name="hr",
        edn="""{:employee/name {:db/valueType :db.type/string}
 :employee/email {:db/valueType :db.type/string}
 :employee/department {:db/valueType :db.type/ref}
 :employee/manager {:db/valueType :db.type/ref}
 :employee/salary {:db/valueType :db.type/double}
 :employee/hired {:db/valueType :db.type/instant}
 :employee/title {:db/valueType :db.type/string}
 :department/name {:db/valueType :db.type/string}
 :department/budget {:db/valueType :db.type/double}}""",
        description="HR management"
    ),
    Schema(
        name="library",
        edn="""{:book/title {:db/valueType :db.type/string :db/fulltext true}
 :book/author {:db/valueType :db.type/string}
 :book/isbn {:db/valueType :db.type/string :db/unique :db.unique/identity}
 :book/published {:db/valueType :db.type/instant}
 :book/genre {:db/valueType :db.type/keyword}
 :book/pages {:db/valueType :db.type/long}
 :loan/book {:db/valueType :db.type/ref}
 :loan/member {:db/valueType :db.type/ref}
 :loan/borrowed {:db/valueType :db.type/instant}
 :loan/due {:db/valueType :db.type/instant}
 :loan/returned {:db/valueType :db.type/instant}
 :member/name {:db/valueType :db.type/string}
 :member/email {:db/valueType :db.type/string}}""",
        description="Library system"
    ),
    Schema(
        name="inventory",
        edn="""{:item/sku {:db/valueType :db.type/string :db/unique :db.unique/identity}
 :item/name {:db/valueType :db.type/string}
 :item/quantity {:db/valueType :db.type/long}
 :item/location {:db/valueType :db.type/ref}
 :item/reorder-point {:db/valueType :db.type/long}
 :item/unit-cost {:db/valueType :db.type/double}
 :location/name {:db/valueType :db.type/string}
 :location/type {:db/valueType :db.type/keyword}}""",
        description="Inventory tracking"
    ),
]


def generate_basic_finds(schema: Schema) -> List[Example]:
    """Generate basic find all queries."""
    examples = []

    if schema.name == "users":
        examples.extend([
            Example(schema, "Find all users", "[:find ?e :where [?e :user/name _]]", "basic"),
            Example(schema, "Get all users", "[:find ?e :where [?e :user/name _]]", "basic"),
            Example(schema, "Show me all users", "[:find ?e :where [?e :user/name _]]", "basic"),
            Example(schema, "List every user", "[:find ?e :where [?e :user/name _]]", "basic"),
            Example(schema, "Find all user names", "[:find ?name :where [_ :user/name ?name]]", "basic"),
            Example(schema, "Get user names and emails", "[:find ?name ?email :where [?e :user/name ?name] [?e :user/email ?email]]", "basic"),
            Example(schema, "Show all user emails", "[:find ?email :where [_ :user/email ?email]]", "basic"),
        ])
    elif schema.name == "ecommerce":
        examples.extend([
            Example(schema, "Find all products", "[:find ?e :where [?e :product/name _]]", "basic"),
            Example(schema, "List all orders", "[:find ?e :where [?e :order/total _]]", "basic"),
            Example(schema, "Get product names and prices", "[:find ?name ?price :where [?e :product/name ?name] [?e :product/price ?price]]", "basic"),
            Example(schema, "Show all product categories", "[:find ?cat :where [_ :product/category ?cat]]", "basic"),
        ])
    elif schema.name == "blog":
        examples.extend([
            Example(schema, "Find all posts", "[:find ?e :where [?e :post/title _]]", "basic"),
            Example(schema, "List all comments", "[:find ?e :where [?e :comment/text _]]", "basic"),
            Example(schema, "Get post titles", "[:find ?title :where [_ :post/title ?title]]", "basic"),
            Example(schema, "Show all authors", "[:find ?e :where [?e :author/name _]]", "basic"),
            Example(schema, "Get author names", "[:find ?name :where [_ :author/name ?name]]", "basic"),
        ])
    elif schema.name == "hr":
        examples.extend([
            Example(schema, "Find all employees", "[:find ?e :where [?e :employee/name _]]", "basic"),
            Example(schema, "List all departments", "[:find ?e :where [?e :department/name _]]", "basic"),
            Example(schema, "Get employee names and titles", "[:find ?name ?title :where [?e :employee/name ?name] [?e :employee/title ?title]]", "basic"),
        ])
    elif schema.name == "library":
        examples.extend([
            Example(schema, "Find all books", "[:find ?e :where [?e :book/title _]]", "basic"),
            Example(schema, "List all members", "[:find ?e :where [?e :member/name _]]", "basic"),
            Example(schema, "Get book titles and authors", "[:find ?title ?author :where [?e :book/title ?title] [?e :book/author ?author]]", "basic"),
            Example(schema, "Show all loans", "[:find ?e :where [?e :loan/book _]]", "basic"),
        ])
    elif schema.name == "inventory":
        examples.extend([
            Example(schema, "Find all items", "[:find ?e :where [?e :item/name _]]", "basic"),
            Example(schema, "List all locations", "[:find ?e :where [?e :location/name _]]", "basic"),
            Example(schema, "Get item names and quantities", "[:find ?name ?qty :where [?e :item/name ?name] [?e :item/quantity ?qty]]", "basic"),
        ])

    return examples


def generate_equality_filters(schema: Schema) -> List[Example]:
    """Generate equality filter queries."""
    examples = []

    if schema.name == "users":
        examples.extend([
            Example(schema, "Find users named John", '[:find ?e :where [?e :user/name "John"]]', "filter"),
            Example(schema, "Find users named Alice", '[:find ?e :where [?e :user/name "Alice"]]', "filter"),
            Example(schema, "Get user with email john@example.com", '[:find ?e :where [?e :user/email "john@example.com"]]', "filter"),
            Example(schema, "Find active users", "[:find ?e :where [?e :user/active true]]", "filter"),
            Example(schema, "Find inactive users", "[:find ?e :where [?e :user/active false]]", "filter"),
            Example(schema, "Find admin users", "[:find ?e :where [?e :user/role :admin]]", "filter"),
            Example(schema, "Find users with role moderator", "[:find ?e :where [?e :user/role :moderator]]", "filter"),
        ])
    elif schema.name == "ecommerce":
        examples.extend([
            Example(schema, "Find products in electronics category", "[:find ?e :where [?e :product/category :electronics]]", "filter"),
            Example(schema, "Find active products", "[:find ?e :where [?e :product/active true]]", "filter"),
            Example(schema, "Find pending orders", "[:find ?e :where [?e :order/status :pending]]", "filter"),
            Example(schema, "Find completed orders", "[:find ?e :where [?e :order/status :completed]]", "filter"),
            Example(schema, "Find shipped orders", "[:find ?e :where [?e :order/status :shipped]]", "filter"),
        ])
    elif schema.name == "blog":
        examples.extend([
            Example(schema, "Find posts tagged with clojure", "[:find ?e :where [?e :post/tags :clojure]]", "filter"),
            Example(schema, "Find posts tagged programming", "[:find ?e :where [?e :post/tags :programming]]", "filter"),
        ])
    elif schema.name == "library":
        examples.extend([
            Example(schema, "Find fiction books", "[:find ?e :where [?e :book/genre :fiction]]", "filter"),
            Example(schema, "Find science fiction books", "[:find ?e :where [?e :book/genre :science-fiction]]", "filter"),
            Example(schema, "Find mystery books", "[:find ?e :where [?e :book/genre :mystery]]", "filter"),
        ])
    elif schema.name == "inventory":
        examples.extend([
            Example(schema, "Find items in warehouse", "[:find ?e :where [?e :item/location ?loc] [?loc :location/type :warehouse]]", "filter"),
        ])

    return examples


def generate_comparison_filters(schema: Schema) -> List[Example]:
    """Generate comparison filter queries."""
    examples = []

    if schema.name == "users":
        examples.extend([
            Example(schema, "Find users older than 30", "[:find ?e :where [?e :user/age ?age] [(> ?age 30)]]", "comparison"),
            Example(schema, "Find users younger than 25", "[:find ?e :where [?e :user/age ?age] [(< ?age 25)]]", "comparison"),
            Example(schema, "Find users aged 18 or older", "[:find ?e :where [?e :user/age ?age] [(>= ?age 18)]]", "comparison"),
            Example(schema, "Find users under 65", "[:find ?e :where [?e :user/age ?age] [(< ?age 65)]]", "comparison"),
            Example(schema, "Find users between 20 and 40", "[:find ?e :where [?e :user/age ?age] [(>= ?age 20)] [(<= ?age 40)]]", "comparison"),
            Example(schema, "Users with age greater than 50", "[:find ?e :where [?e :user/age ?age] [(> ?age 50)]]", "comparison"),
        ])
    elif schema.name == "ecommerce":
        examples.extend([
            Example(schema, "Find products under $50", "[:find ?e :where [?e :product/price ?p] [(< ?p 50.0)]]", "comparison"),
            Example(schema, "Find products over $100", "[:find ?e :where [?e :product/price ?p] [(> ?p 100.0)]]", "comparison"),
            Example(schema, "Find products priced between $10 and $50", "[:find ?e :where [?e :product/price ?p] [(>= ?p 10.0)] [(<= ?p 50.0)]]", "comparison"),
            Example(schema, "Find products with low stock", "[:find ?e :where [?e :product/stock ?s] [(< ?s 10)]]", "comparison"),
            Example(schema, "Find products with stock below 5", "[:find ?e :where [?e :product/stock ?s] [(< ?s 5)]]", "comparison"),
            Example(schema, "Find orders over $1000", "[:find ?e :where [?e :order/total ?t] [(> ?t 1000.0)]]", "comparison"),
            Example(schema, "Orders with total greater than 500", "[:find ?e :where [?e :order/total ?t] [(> ?t 500.0)]]", "comparison"),
        ])
    elif schema.name == "blog":
        examples.extend([
            Example(schema, "Find posts with more than 1000 views", "[:find ?e :where [?e :post/views ?v] [(> ?v 1000)]]", "comparison"),
            Example(schema, "Find popular posts with over 5000 views", "[:find ?e :where [?e :post/views ?v] [(> ?v 5000)]]", "comparison"),
            Example(schema, "Posts with less than 100 views", "[:find ?e :where [?e :post/views ?v] [(< ?v 100)]]", "comparison"),
        ])
    elif schema.name == "hr":
        examples.extend([
            Example(schema, "Find employees earning over 100000", "[:find ?e :where [?e :employee/salary ?s] [(> ?s 100000.0)]]", "comparison"),
            Example(schema, "Find employees with salary under 50000", "[:find ?e :where [?e :employee/salary ?s] [(< ?s 50000.0)]]", "comparison"),
            Example(schema, "Employees earning between 60000 and 80000", "[:find ?e :where [?e :employee/salary ?s] [(>= ?s 60000.0)] [(<= ?s 80000.0)]]", "comparison"),
            Example(schema, "Find departments with budget over 1 million", "[:find ?e :where [?e :department/budget ?b] [(> ?b 1000000.0)]]", "comparison"),
        ])
    elif schema.name == "library":
        examples.extend([
            Example(schema, "Find books with more than 500 pages", "[:find ?e :where [?e :book/pages ?p] [(> ?p 500)]]", "comparison"),
            Example(schema, "Find short books under 200 pages", "[:find ?e :where [?e :book/pages ?p] [(< ?p 200)]]", "comparison"),
        ])
    elif schema.name == "inventory":
        examples.extend([
            Example(schema, "Find items below reorder point", "[:find ?e :where [?e :item/quantity ?q] [?e :item/reorder-point ?r] [(< ?q ?r)]]", "comparison"),
            Example(schema, "Find items with quantity over 100", "[:find ?e :where [?e :item/quantity ?q] [(> ?q 100)]]", "comparison"),
            Example(schema, "Find expensive items over $50 unit cost", "[:find ?e :where [?e :item/unit-cost ?c] [(> ?c 50.0)]]", "comparison"),
        ])

    return examples


def generate_string_operations(schema: Schema) -> List[Example]:
    """Generate string operation queries."""
    examples = []

    if schema.name == "users":
        examples.extend([
            Example(schema, "Find users with gmail addresses", '[:find ?e :where [?e :user/email ?email] [(clojure.string/includes? ?email "gmail")]]', "string"),
            Example(schema, "Find users with yahoo email", '[:find ?e :where [?e :user/email ?email] [(clojure.string/includes? ?email "yahoo")]]', "string"),
            Example(schema, "Find users with company email", '[:find ?e :where [?e :user/email ?email] [(clojure.string/ends-with? ?email "company.com")]]', "string"),
            Example(schema, "Find users whose name starts with A", '[:find ?e :where [?e :user/name ?name] [(clojure.string/starts-with? ?name "A")]]', "string"),
            Example(schema, "Find users with Dr. title", '[:find ?e :where [?e :user/name ?name] [(clojure.string/starts-with? ?name "Dr.")]]', "string"),
        ])
    elif schema.name == "ecommerce":
        examples.extend([
            Example(schema, "Find products containing laptop in name", '[:find ?e :where [?e :product/name ?name] [(clojure.string/includes? ?name "laptop")]]', "string"),
            Example(schema, "Find products with Pro in name", '[:find ?e :where [?e :product/name ?name] [(clojure.string/includes? ?name "Pro")]]', "string"),
        ])
    elif schema.name == "hr":
        examples.extend([
            Example(schema, "Find employees with corporate email", '[:find ?e :where [?e :employee/email ?email] [(clojure.string/ends-with? ?email "corp.com")]]', "string"),
            Example(schema, "Find senior employees by title", '[:find ?e :where [?e :employee/title ?title] [(clojure.string/starts-with? ?title "Senior")]]', "string"),
        ])
    elif schema.name == "library":
        examples.extend([
            Example(schema, "Find books by author Stephen", '[:find ?e :where [?e :book/author ?author] [(clojure.string/starts-with? ?author "Stephen")]]', "string"),
        ])

    return examples


def generate_date_operations(schema: Schema) -> List[Example]:
    """Generate date comparison queries."""
    examples = []

    if schema.name == "users":
        examples.extend([
            Example(schema, "Find users created after 2024", '[:find ?e :where [?e :user/created ?d] [(> ?d #inst "2024-01-01")]]', "date"),
            Example(schema, "Find users who joined in 2023", '[:find ?e :where [?e :user/created ?d] [(>= ?d #inst "2023-01-01")] [(< ?d #inst "2024-01-01")]]', "date"),
            Example(schema, "Find users created before 2020", '[:find ?e :where [?e :user/created ?d] [(< ?d #inst "2020-01-01")]]', "date"),
        ])
    elif schema.name == "ecommerce":
        examples.extend([
            Example(schema, "Find orders from 2024", '[:find ?e :where [?e :order/date ?d] [(>= ?d #inst "2024-01-01")]]', "date"),
            Example(schema, "Find orders before December 2023", '[:find ?e :where [?e :order/date ?d] [(< ?d #inst "2023-12-01")]]', "date"),
            Example(schema, "Find recent orders after March 2024", '[:find ?e :where [?e :order/date ?d] [(> ?d #inst "2024-03-01")]]', "date"),
        ])
    elif schema.name == "blog":
        examples.extend([
            Example(schema, "Find posts published in 2024", '[:find ?e :where [?e :post/published ?d] [(>= ?d #inst "2024-01-01")]]', "date"),
            Example(schema, "Find old posts before 2020", '[:find ?e :where [?e :post/published ?d] [(< ?d #inst "2020-01-01")]]', "date"),
        ])
    elif schema.name == "hr":
        examples.extend([
            Example(schema, "Find employees hired after 2022", '[:find ?e :where [?e :employee/hired ?d] [(> ?d #inst "2022-01-01")]]', "date"),
            Example(schema, "Find employees hired before 2015", '[:find ?e :where [?e :employee/hired ?d] [(< ?d #inst "2015-01-01")]]', "date"),
        ])
    elif schema.name == "library":
        examples.extend([
            Example(schema, "Find books published after 2000", '[:find ?e :where [?e :book/published ?d] [(> ?d #inst "2000-01-01")]]', "date"),
            Example(schema, "Find overdue loans", '[:find ?e :in $ ?now :where [?e :loan/due ?due] [(< ?due ?now)] (not [?e :loan/returned _])]', "date"),
            Example(schema, "Find loans due before today", '[:find ?e :in $ ?now :where [?e :loan/due ?due] [(< ?due ?now)]]', "date"),
        ])

    return examples


def generate_aggregations(schema: Schema) -> List[Example]:
    """Generate aggregation queries."""
    examples = []

    if schema.name == "users":
        examples.extend([
            Example(schema, "Count all users", "[:find (count ?e) :where [?e :user/name _]]", "aggregation"),
            Example(schema, "Count active users", "[:find (count ?e) :where [?e :user/active true]]", "aggregation"),
            Example(schema, "Average user age", "[:find (avg ?age) :where [?e :user/age ?age]]", "aggregation"),
            Example(schema, "Find oldest user age", "[:find (max ?age) :where [?e :user/age ?age]]", "aggregation"),
            Example(schema, "Find youngest user age", "[:find (min ?age) :where [?e :user/age ?age]]", "aggregation"),
            Example(schema, "Median user age", "[:find (median ?age) :where [?e :user/age ?age]]", "aggregation"),
            Example(schema, "Count users by role", "[:find ?role (count ?e) :where [?e :user/role ?role]]", "aggregation"),
            Example(schema, "How many users are there", "[:find (count ?e) :where [?e :user/name _]]", "aggregation"),
        ])
    elif schema.name == "ecommerce":
        examples.extend([
            Example(schema, "Count all products", "[:find (count ?e) :where [?e :product/name _]]", "aggregation"),
            Example(schema, "Count all orders", "[:find (count ?e) :where [?e :order/total _]]", "aggregation"),
            Example(schema, "Average product price", "[:find (avg ?p) :where [?e :product/price ?p]]", "aggregation"),
            Example(schema, "Total order revenue", "[:find (sum ?t) :where [?e :order/total ?t]]", "aggregation"),
            Example(schema, "Average order total", "[:find (avg ?t) :where [?e :order/total ?t]]", "aggregation"),
            Example(schema, "Maximum product price", "[:find (max ?p) :where [?e :product/price ?p]]", "aggregation"),
            Example(schema, "Count products by category", "[:find ?cat (count ?e) :where [?e :product/category ?cat]]", "aggregation"),
            Example(schema, "Count orders by status", "[:find ?status (count ?e) :where [?e :order/status ?status]]", "aggregation"),
            Example(schema, "Total stock across all products", "[:find (sum ?s) :where [?e :product/stock ?s]]", "aggregation"),
        ])
    elif schema.name == "blog":
        examples.extend([
            Example(schema, "Count all posts", "[:find (count ?e) :where [?e :post/title _]]", "aggregation"),
            Example(schema, "Total views across all posts", "[:find (sum ?v) :where [?e :post/views ?v]]", "aggregation"),
            Example(schema, "Average post views", "[:find (avg ?v) :where [?e :post/views ?v]]", "aggregation"),
            Example(schema, "Most viewed post views count", "[:find (max ?v) :where [?e :post/views ?v]]", "aggregation"),
            Example(schema, "Count posts by tag", "[:find ?tag (count ?e) :where [?e :post/tags ?tag]]", "aggregation"),
            Example(schema, "Count comments", "[:find (count ?e) :where [?e :comment/text _]]", "aggregation"),
        ])
    elif schema.name == "hr":
        examples.extend([
            Example(schema, "Count employees", "[:find (count ?e) :where [?e :employee/name _]]", "aggregation"),
            Example(schema, "Average salary", "[:find (avg ?s) :where [?e :employee/salary ?s]]", "aggregation"),
            Example(schema, "Total salary expense", "[:find (sum ?s) :where [?e :employee/salary ?s]]", "aggregation"),
            Example(schema, "Highest salary", "[:find (max ?s) :where [?e :employee/salary ?s]]", "aggregation"),
            Example(schema, "Lowest salary", "[:find (min ?s) :where [?e :employee/salary ?s]]", "aggregation"),
            Example(schema, "Median salary", "[:find (median ?s) :where [?e :employee/salary ?s]]", "aggregation"),
            Example(schema, "Count departments", "[:find (count ?e) :where [?e :department/name _]]", "aggregation"),
        ])
    elif schema.name == "library":
        examples.extend([
            Example(schema, "Count all books", "[:find (count ?e) :where [?e :book/title _]]", "aggregation"),
            Example(schema, "Average book pages", "[:find (avg ?p) :where [?e :book/pages ?p]]", "aggregation"),
            Example(schema, "Count books by genre", "[:find ?genre (count ?e) :where [?e :book/genre ?genre]]", "aggregation"),
            Example(schema, "Count active loans", "[:find (count ?e) :where [?e :loan/book _] (not [?e :loan/returned _])]", "aggregation"),
        ])
    elif schema.name == "inventory":
        examples.extend([
            Example(schema, "Count all items", "[:find (count ?e) :where [?e :item/name _]]", "aggregation"),
            Example(schema, "Total inventory value", "[:find (sum ?v) :where [?e :item/quantity ?q] [?e :item/unit-cost ?c] [(* ?q ?c) ?v]]", "aggregation"),
            Example(schema, "Total quantity across all items", "[:find (sum ?q) :where [?e :item/quantity ?q]]", "aggregation"),
        ])

    return examples


def generate_joins(schema: Schema) -> List[Example]:
    """Generate join queries."""
    examples = []

    if schema.name == "users":
        examples.extend([
            Example(schema, "Find users and their friends names", "[:find ?name ?friend-name :where [?e :user/name ?name] [?e :user/friends ?f] [?f :user/name ?friend-name]]", "join"),
            Example(schema, "Find user pairs who are friends", "[:find ?n1 ?n2 :where [?e1 :user/name ?n1] [?e1 :user/friends ?e2] [?e2 :user/name ?n2]]", "join"),
        ])
    elif schema.name == "ecommerce":
        examples.extend([
            Example(schema, "Find orders with customer names", "[:find ?order ?name :where [?order :order/user ?user] [?user :user/name ?name]]", "join"),
            Example(schema, "Find product names in each order", "[:find ?order ?product-name :where [?order :order/products ?product] [?product :product/name ?product-name]]", "join"),
            Example(schema, "Orders with total and customer email", "[:find ?total ?email :where [?o :order/total ?total] [?o :order/user ?u] [?u :user/email ?email]]", "join"),
        ])
    elif schema.name == "blog":
        examples.extend([
            Example(schema, "Find posts with author names", "[:find ?title ?author-name :where [?p :post/title ?title] [?p :post/author ?a] [?a :author/name ?author-name]]", "join"),
            Example(schema, "Find comments with post titles", "[:find ?comment ?title :where [?c :comment/text ?comment] [?c :comment/post ?p] [?p :post/title ?title]]", "join"),
            Example(schema, "Find comments with author and post info", "[:find ?comment ?author-name ?post-title :where [?c :comment/text ?comment] [?c :comment/author ?a] [?a :author/name ?author-name] [?c :comment/post ?p] [?p :post/title ?post-title]]", "join"),
        ])
    elif schema.name == "hr":
        examples.extend([
            Example(schema, "Find employees with department names", "[:find ?emp-name ?dept-name :where [?e :employee/name ?emp-name] [?e :employee/department ?d] [?d :department/name ?dept-name]]", "join"),
            Example(schema, "Find employees with their manager names", "[:find ?emp ?mgr :where [?e :employee/name ?emp] [?e :employee/manager ?m] [?m :employee/name ?mgr]]", "join"),
            Example(schema, "Find employees and manager in same department", "[:find ?emp ?mgr :where [?e :employee/name ?emp] [?e :employee/department ?d] [?e :employee/manager ?m] [?m :employee/name ?mgr] [?m :employee/department ?d]]", "join"),
        ])
    elif schema.name == "library":
        examples.extend([
            Example(schema, "Find loans with book titles", "[:find ?loan ?title :where [?loan :loan/book ?b] [?b :book/title ?title]]", "join"),
            Example(schema, "Find loans with member names", "[:find ?loan ?name :where [?loan :loan/member ?m] [?m :member/name ?name]]", "join"),
            Example(schema, "Find who borrowed which book", "[:find ?member-name ?book-title :where [?l :loan/member ?m] [?m :member/name ?member-name] [?l :loan/book ?b] [?b :book/title ?book-title]]", "join"),
        ])
    elif schema.name == "inventory":
        examples.extend([
            Example(schema, "Find items with location names", "[:find ?item-name ?loc-name :where [?i :item/name ?item-name] [?i :item/location ?l] [?l :location/name ?loc-name]]", "join"),
            Example(schema, "Items in warehouse locations", "[:find ?item-name :where [?i :item/name ?item-name] [?i :item/location ?l] [?l :location/type :warehouse]]", "join"),
        ])

    return examples


def generate_negations(schema: Schema) -> List[Example]:
    """Generate negation queries."""
    examples = []

    if schema.name == "users":
        examples.extend([
            Example(schema, "Find users without email", "[:find ?e :where [?e :user/name _] (not [?e :user/email _])]", "negation"),
            Example(schema, "Find inactive users", "[:find ?e :where [?e :user/name _] (not [?e :user/active true])]", "negation"),
            Example(schema, "Find users with no friends", "[:find ?e :where [?e :user/name _] (not [?e :user/friends _])]", "negation"),
            Example(schema, "Find non-admin users", "[:find ?e :where [?e :user/name _] (not [?e :user/role :admin])]", "negation"),
        ])
    elif schema.name == "ecommerce":
        examples.extend([
            Example(schema, "Find products not in any category", "[:find ?e :where [?e :product/name _] (not [?e :product/category _])]", "negation"),
            Example(schema, "Find orders without products", "[:find ?e :where [?e :order/total _] (not [?e :order/products _])]", "negation"),
            Example(schema, "Find non-shipped orders", "[:find ?e :where [?e :order/status _] (not [?e :order/status :shipped])]", "negation"),
        ])
    elif schema.name == "blog":
        examples.extend([
            Example(schema, "Find posts without tags", "[:find ?e :where [?e :post/title _] (not [?e :post/tags _])]", "negation"),
            Example(schema, "Find posts without comments", "[:find ?e :where [?e :post/title _] (not [?c :comment/post ?e])]", "negation"),
            Example(schema, "Find authors without posts", "[:find ?e :where [?e :author/name _] (not [?p :post/author ?e])]", "negation"),
        ])
    elif schema.name == "hr":
        examples.extend([
            Example(schema, "Find employees without manager", "[:find ?e :where [?e :employee/name _] (not [?e :employee/manager _])]", "negation"),
            Example(schema, "Find employees not in any department", "[:find ?e :where [?e :employee/name _] (not [?e :employee/department _])]", "negation"),
        ])
    elif schema.name == "library":
        examples.extend([
            Example(schema, "Find books never borrowed", "[:find ?e :where [?e :book/title _] (not [?l :loan/book ?e])]", "negation"),
            Example(schema, "Find unreturned loans", "[:find ?e :where [?e :loan/book _] (not [?e :loan/returned _])]", "negation"),
            Example(schema, "Find members with no loans", "[:find ?e :where [?e :member/name _] (not [?l :loan/member ?e])]", "negation"),
        ])
    elif schema.name == "inventory":
        examples.extend([
            Example(schema, "Find items without location", "[:find ?e :where [?e :item/name _] (not [?e :item/location _])]", "negation"),
        ])

    return examples


def generate_pull_queries(schema: Schema) -> List[Example]:
    """Generate pull syntax queries."""
    examples = []

    if schema.name == "users":
        examples.extend([
            Example(schema, "Get full user details", "[:find (pull ?e [*]) :where [?e :user/name _]]", "pull"),
            Example(schema, "Get user name and email only", "[:find (pull ?e [:user/name :user/email]) :where [?e :user/name _]]", "pull"),
            Example(schema, "Get active users with all attributes", "[:find (pull ?e [*]) :where [?e :user/active true]]", "pull"),
            Example(schema, "Get users with their friends", "[:find (pull ?e [:user/name {:user/friends [:user/name]}]) :where [?e :user/name _]]", "pull"),
        ])
    elif schema.name == "ecommerce":
        examples.extend([
            Example(schema, "Get full product details", "[:find (pull ?e [*]) :where [?e :product/name _]]", "pull"),
            Example(schema, "Get order with products", "[:find (pull ?e [:order/total {:order/products [:product/name :product/price]}]) :where [?e :order/total _]]", "pull"),
            Example(schema, "Get order with customer info", "[:find (pull ?e [:order/total :order/date {:order/user [:user/name :user/email]}]) :where [?e :order/total _]]", "pull"),
        ])
    elif schema.name == "blog":
        examples.extend([
            Example(schema, "Get full post details", "[:find (pull ?e [*]) :where [?e :post/title _]]", "pull"),
            Example(schema, "Get posts with author info", "[:find (pull ?e [:post/title {:post/author [:author/name]}]) :where [?e :post/title _]]", "pull"),
            Example(schema, "Get post with comments", "[:find (pull ?p [:post/title :post/body]) (pull ?c [:comment/text]) :where [?p :post/title _] [?c :comment/post ?p]]", "pull"),
        ])
    elif schema.name == "hr":
        examples.extend([
            Example(schema, "Get employee with department", "[:find (pull ?e [:employee/name :employee/title {:employee/department [:department/name]}]) :where [?e :employee/name _]]", "pull"),
            Example(schema, "Get full employee details", "[:find (pull ?e [*]) :where [?e :employee/name _]]", "pull"),
        ])
    elif schema.name == "library":
        examples.extend([
            Example(schema, "Get full book details", "[:find (pull ?e [*]) :where [?e :book/title _]]", "pull"),
            Example(schema, "Get loan with book and member", "[:find (pull ?l [:loan/borrowed :loan/due {:loan/book [:book/title]} {:loan/member [:member/name]}]) :where [?l :loan/book _]]", "pull"),
        ])

    return examples


def generate_fulltext_queries(schema: Schema) -> List[Example]:
    """Generate full-text search queries (Datalevin-specific)."""
    examples = []

    if schema.name == "blog":
        examples.extend([
            Example(schema, "Search posts for clojure", '[:find ?e ?a ?v :where [(fulltext $ "clojure") [[?e ?a ?v]]]]', "fulltext"),
            Example(schema, "Search post titles for programming", '[:find ?e ?a ?v :where [(fulltext $ :post/title "programming") [[?e ?a ?v]]]]', "fulltext"),
            Example(schema, "Search posts for machine learning", '[:find ?e ?a ?v :where [(fulltext $ "machine learning") [[?e ?a ?v]]]]', "fulltext"),
            Example(schema, "Search for exact phrase functional programming", '[:find ?e ?a ?v :where [(fulltext $ {:phrase "functional programming"}) [[?e ?a ?v]]]]', "fulltext"),
            Example(schema, "Search for clojure but not java", '[:find ?e ?a ?v :where [(fulltext $ [:and "clojure" [:not "java"]]) [[?e ?a ?v]]]]', "fulltext"),
            Example(schema, "Search for python or javascript", '[:find ?e ?a ?v :where [(fulltext $ [:or "python" "javascript"]) [[?e ?a ?v]]]]', "fulltext"),
            Example(schema, "Search author bios for expert", '[:find ?e ?a ?v :where [(fulltext $ :author/bio "expert") [[?e ?a ?v]]]]', "fulltext"),
        ])
    elif schema.name == "library":
        examples.extend([
            Example(schema, "Search books for science fiction", '[:find ?e ?a ?v :where [(fulltext $ "science fiction") [[?e ?a ?v]]]]', "fulltext"),
            Example(schema, "Search book titles for adventure", '[:find ?e ?a ?v :where [(fulltext $ :book/title "adventure") [[?e ?a ?v]]]]', "fulltext"),
        ])

    return examples


def generate_or_queries(schema: Schema) -> List[Example]:
    """Generate OR clause queries."""
    examples = []

    if schema.name == "users":
        examples.extend([
            Example(schema, "Find active or admin users", "[:find ?e :where (or [?e :user/active true] [?e :user/role :admin])]", "or"),
            Example(schema, "Find users named John or Jane", '[:find ?e :where (or [?e :user/name "John"] [?e :user/name "Jane"])]', "or"),
            Example(schema, "Find young or old users", "[:find ?e :where [?e :user/age ?age] (or [(< ?age 20)] [(> ?age 60)])]", "or"),
        ])
    elif schema.name == "ecommerce":
        examples.extend([
            Example(schema, "Find pending or processing orders", "[:find ?e :where (or [?e :order/status :pending] [?e :order/status :processing])]", "or"),
            Example(schema, "Find electronics or clothing products", "[:find ?e :where (or [?e :product/category :electronics] [?e :product/category :clothing])]", "or"),
        ])
    elif schema.name == "blog":
        examples.extend([
            Example(schema, "Find posts tagged clojure or java", "[:find ?e :where (or [?e :post/tags :clojure] [?e :post/tags :java])]", "or"),
        ])
    elif schema.name == "hr":
        examples.extend([
            Example(schema, "Find engineers or managers", '[:find ?e :where (or [?e :employee/title "Engineer"] [?e :employee/title "Manager"])]', "or"),
        ])

    return examples


def generate_return_formats(schema: Schema) -> List[Example]:
    """Generate queries with different return formats."""
    examples = []

    if schema.name == "users":
        examples.extend([
            Example(schema, "Get all unique user names as a list", "[:find [?name ...] :where [_ :user/name ?name]]", "return-format"),
            Example(schema, "Get single user by email", '[:find ?e . :where [?e :user/email "admin@example.com"]]', "return-format"),
            Example(schema, "Get first user name and age", "[:find [?name ?age] :where [?e :user/name ?name] [?e :user/age ?age]]", "return-format"),
            Example(schema, "Get users as maps with keys", "[:find ?e ?name ?email :keys id name email :where [?e :user/name ?name] [?e :user/email ?email]]", "return-format"),
        ])
    elif schema.name == "ecommerce":
        examples.extend([
            Example(schema, "Get all unique categories as a list", "[:find [?cat ...] :where [_ :product/category ?cat]]", "return-format"),
            Example(schema, "Get all unique order statuses", "[:find [?status ...] :where [_ :order/status ?status]]", "return-format"),
        ])
    elif schema.name == "blog":
        examples.extend([
            Example(schema, "Get all unique tags", "[:find [?tag ...] :where [_ :post/tags ?tag]]", "return-format"),
        ])

    return examples


def generate_with_inputs(schema: Schema) -> List[Example]:
    """Generate queries that use :in clause for inputs."""
    examples = []

    if schema.name == "users":
        examples.extend([
            Example(schema, "Find user by given name", "[:find ?e :in $ ?name :where [?e :user/name ?name]]", "input"),
            Example(schema, "Find users by given role", "[:find ?e :in $ ?role :where [?e :user/role ?role]]", "input"),
            Example(schema, "Find users in age range", "[:find ?e :in $ ?min ?max :where [?e :user/age ?age] [(>= ?age ?min)] [(<= ?age ?max)]]", "input"),
            Example(schema, "Find users with any of these emails", "[:find ?e :in $ [?email ...] :where [?e :user/email ?email]]", "input"),
        ])
    elif schema.name == "ecommerce":
        examples.extend([
            Example(schema, "Find products by given category", "[:find ?e :in $ ?cat :where [?e :product/category ?cat]]", "input"),
            Example(schema, "Find products in price range", "[:find ?e :in $ ?min ?max :where [?e :product/price ?p] [(>= ?p ?min)] [(<= ?p ?max)]]", "input"),
            Example(schema, "Find orders by status", "[:find ?e :in $ ?status :where [?e :order/status ?status]]", "input"),
        ])
    elif schema.name == "hr":
        examples.extend([
            Example(schema, "Find employees by department", "[:find ?e :in $ ?dept :where [?e :employee/department ?d] [?d :department/name ?dept]]", "input"),
            Example(schema, "Find employees in salary range", "[:find ?e :in $ ?min ?max :where [?e :employee/salary ?s] [(>= ?s ?min)] [(<= ?s ?max)]]", "input"),
        ])
    elif schema.name == "library":
        examples.extend([
            Example(schema, "Find books by genre", "[:find ?e :in $ ?genre :where [?e :book/genre ?genre]]", "input"),
            Example(schema, "Find books by author name", "[:find ?e :in $ ?author :where [?e :book/author ?author]]", "input"),
        ])

    return examples


def generate_complex_queries(schema: Schema) -> List[Example]:
    """Generate more complex combined queries."""
    examples = []

    if schema.name == "users":
        examples.extend([
            Example(schema, "Find active adult users with gmail", '[:find ?e ?name :where [?e :user/name ?name] [?e :user/active true] [?e :user/age ?age] [(>= ?age 18)] [?e :user/email ?email] [(clojure.string/includes? ?email "gmail")]]', "complex"),
            Example(schema, "Count active users by role", "[:find ?role (count ?e) :where [?e :user/active true] [?e :user/role ?role]]", "complex"),
        ])
    elif schema.name == "ecommerce":
        examples.extend([
            Example(schema, "Find high-value completed orders with customer info", "[:find ?total ?name ?email :where [?o :order/total ?total] [(> ?total 500.0)] [?o :order/status :completed] [?o :order/user ?u] [?u :user/name ?name] [?u :user/email ?email]]", "complex"),
            Example(schema, "Find low stock active products", "[:find ?name ?stock :where [?e :product/name ?name] [?e :product/stock ?stock] [(< ?stock 10)] [?e :product/active true]]", "complex"),
            Example(schema, "Total revenue by category", "[:find ?cat (sum ?total) :where [?o :order/products ?p] [?p :product/category ?cat] [?o :order/total ?total]]", "complex"),
        ])
    elif schema.name == "blog":
        examples.extend([
            Example(schema, "Find popular posts from 2024 by author", "[:find ?title ?author-name ?views :where [?p :post/title ?title] [?p :post/views ?views] [(> ?views 1000)] [?p :post/published ?d] [(>= ?d #inst \"2024-01-01\")] [?p :post/author ?a] [?a :author/name ?author-name]]", "complex"),
            Example(schema, "Count comments per post", "[:find ?title (count ?c) :where [?p :post/title ?title] [?c :comment/post ?p]]", "complex"),
        ])
    elif schema.name == "hr":
        examples.extend([
            Example(schema, "Find high earners by department", "[:find ?name ?dept-name ?salary :where [?e :employee/name ?name] [?e :employee/salary ?salary] [(> ?salary 100000.0)] [?e :employee/department ?d] [?d :department/name ?dept-name]]", "complex"),
            Example(schema, "Average salary by department", "[:find ?dept-name (avg ?salary) :where [?e :employee/salary ?salary] [?e :employee/department ?d] [?d :department/name ?dept-name]]", "complex"),
            Example(schema, "Count employees by department", "[:find ?dept-name (count ?e) :where [?e :employee/department ?d] [?d :department/name ?dept-name]]", "complex"),
        ])
    elif schema.name == "library":
        examples.extend([
            Example(schema, "Find overdue loans with book and member info", "[:find ?member-name ?book-title ?due :in $ ?now :where [?l :loan/due ?due] [(< ?due ?now)] (not [?l :loan/returned _]) [?l :loan/member ?m] [?m :member/name ?member-name] [?l :loan/book ?b] [?b :book/title ?book-title]]", "complex"),
            Example(schema, "Count loans per member", "[:find ?name (count ?l) :where [?l :loan/member ?m] [?m :member/name ?name]]", "complex"),
        ])
    elif schema.name == "inventory":
        examples.extend([
            Example(schema, "Find items needing reorder with location", "[:find ?name ?qty ?reorder ?loc-name :where [?i :item/name ?name] [?i :item/quantity ?qty] [?i :item/reorder-point ?reorder] [(< ?qty ?reorder)] [?i :item/location ?l] [?l :location/name ?loc-name]]", "complex"),
        ])

    return examples


def generate_all_examples() -> List[Example]:
    """Generate all training examples."""
    examples = []

    generators = [
        generate_basic_finds,
        generate_equality_filters,
        generate_comparison_filters,
        generate_string_operations,
        generate_date_operations,
        generate_aggregations,
        generate_joins,
        generate_negations,
        generate_pull_queries,
        generate_fulltext_queries,
        generate_or_queries,
        generate_return_formats,
        generate_with_inputs,
        generate_complex_queries,
    ]

    for schema in SCHEMAS:
        for generator in generators:
            examples.extend(generator(schema))

    return examples


def format_example(ex: Example) -> str:
    """Format an example for MLX fine-tuning."""
    return json.dumps({
        "text": f"<|user|>Schema: {ex.schema.edn}\n\n{ex.natural}<|assistant|>{ex.query}"
    })


def main():
    print("Generating Datalevin NLQ training data...")

    # Generate all examples
    examples = generate_all_examples()
    print(f"Generated {len(examples)} examples")

    # Count by category
    categories = {}
    for ex in examples:
        categories[ex.category] = categories.get(ex.category, 0) + 1

    print("\nExamples by category:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")

    # Shuffle examples
    random.seed(42)
    random.shuffle(examples)

    # Split into train/valid
    split_idx = int(len(examples) * (1 - VALID_RATIO))
    train_examples = examples[:split_idx]
    valid_examples = examples[split_idx:]

    print(f"\nTrain examples: {len(train_examples)}")
    print(f"Valid examples: {len(valid_examples)}")

    # Write files
    with open(TRAIN_FILE, 'w') as f:
        for ex in train_examples:
            f.write(format_example(ex) + '\n')

    with open(VALID_FILE, 'w') as f:
        for ex in valid_examples:
            f.write(format_example(ex) + '\n')

    print(f"\nWrote training data to:")
    print(f"  {TRAIN_FILE}")
    print(f"  {VALID_FILE}")


if __name__ == "__main__":
    main()
