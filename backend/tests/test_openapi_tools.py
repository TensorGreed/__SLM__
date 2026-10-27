"""OpenAPI 3.0 / 3.1 -> tool conversion."""

from __future__ import annotations

import unittest

from app.services import json_schema_bundle as jsb
from app.services.openapi_tools import openapi_to_tools

DOC = {
    "openapi": "3.0.3",
    "info": {"title": "Deploy API", "version": "1"},
    "paths": {
        "/services": {
            "get": {"operationId": "listServices", "summary": "List services", "parameters": [
                {"name": "limit", "in": "query", "schema": {"type": "integer", "minimum": 1, "example": 5}},
            ]},
            "post": {"summary": "Create", "requestBody": {"required": True, "content": {"application/json": {"schema": {"$ref": "#/components/schemas/NewService"}}}}},
        },
        "/services/{name}": {
            "parameters": [{"name": "name", "in": "path", "required": True, "schema": {"type": "string"}}],
            "get": {"operationId": "getService", "description": "Fetch one."},
            "delete": {"operationId": "getService", "parameters": [{"name": "X-Trace", "in": "header", "schema": {"type": "string"}}, {"name": "Accept", "in": "header", "schema": {"type": "string"}}]},
        },
    },
    "components": {"schemas": {
        "NewService": {"type": "object", "required": ["name"], "properties": {
            "name": {"type": "string", "example": "api"},
            "owner": {"type": "string", "nullable": True},
            "tier": {"$ref": "#/components/schemas/Tier"},
        }},
        "Tier": {"type": "string", "enum": ["free", "pro"]},
        "Unused": {"type": "object"},
    }},
}


class OpenApiToolsTests(unittest.TestCase):
    def setUp(self):
        self.tools, self.warnings = openapi_to_tools(DOC)
        self.by = {t["name"]: t for t in self.tools}

    def test_names_dedupe_and_annotations(self):
        self.assertEqual(sorted(self.by), ["getService", "getService_2", "listServices", "post_services"])
        self.assertEqual(self.by["listServices"]["annotations"], {"readOnlyHint": True})
        self.assertEqual(self.by["getService_2"]["annotations"]["destructiveHint"], True)
        self.assertFalse(self.by["post_services"]["annotations"]["readOnlyHint"])

    def test_schemas(self):
        get = self.by["getService"]["inputSchema"]
        self.assertEqual(get["required"], ["name"])
        self.assertEqual(self.by["getService"]["description"], "Fetch one.")
        self.assertNotIn("example", self.by["listServices"]["inputSchema"]["properties"]["limit"])
        self.assertEqual(list(self.by["getService_2"]["inputSchema"]["properties"]), ["name", "X-Trace"])  # Accept ignored
        post = self.by["post_services"]["inputSchema"]
        self.assertEqual(post["required"], ["body"])
        self.assertEqual(sorted(post["$defs"]), ["NewService", "Tier"])  # Unused is not copied
        self.assertEqual(post["properties"]["body"], {"$ref": "#/$defs/NewService"})
        self.assertEqual(post["$defs"]["NewService"]["properties"]["owner"]["type"], ["string", "null"])
        self.assertNotIn("example", post["$defs"]["NewService"]["properties"]["name"])

    def test_every_schema_validates(self):
        for t in self.tools:
            jsb.validate_bundle({"tool.json": t["inputSchema"]}, "tool.json")
        post = jsb.validate_bundle({"tool.json": self.by["post_services"]["inputSchema"]}, "tool.json")
        self.assertEqual(post.validate({"body": {"name": "a", "tier": "free", "owner": None}}), [])
        self.assertTrue(post.validate({"body": {"name": "a", "tier": "gold"}}))
        self.assertTrue(post.validate({}))

    def test_31_and_limits(self):
        tools, _ = openapi_to_tools({"openapi": "3.1.0", "paths": {"/a b/{id}": {"get": {"parameters": [{"name": "id", "in": "path", "required": True, "schema": {"type": ["string", "null"]}}]}}}})
        self.assertEqual(tools[0]["name"], "get_a_b_id")
        many = {f"/p{i}": {"get": {}} for i in range(1005)}
        tools, warnings = openapi_to_tools({"openapi": "3.0.0", "paths": many})
        self.assertEqual(len(tools), 1000)
        self.assertTrue(any("1000" in w for w in warnings))
        with self.assertRaises(ValueError):
            openapi_to_tools({"swagger": "2.0", "paths": {}})
        with self.assertRaises(ValueError):
            openapi_to_tools({"openapi": "3.0.0", "info": {"d": "x" * (5 * 1024 * 1024)}, "paths": {"/a": {"get": {}}}})

    def test_recursive_ref_and_bad_ref(self):
        doc = {"openapi": "3.0.0", "paths": {"/t": {"post": {"operationId": "t", "requestBody": {"content": {"application/json": {"schema": {"$ref": "#/components/schemas/Node"}}}}}}},
               "components": {"schemas": {"Node": {"type": "object", "properties": {"next": {"$ref": "#/components/schemas/Node"}}}}}}
        tools, _ = openapi_to_tools(doc)
        self.assertEqual(list(tools[0]["inputSchema"]["$defs"]), ["Node"])

    def test_long_names_and_description_cap(self):
        doc = {"openapi": "3.0.0", "paths": {"/a": {"get": {"operationId": "x" * 80 + " bad!", "description": "d" * 3000}}}}
        tools, _ = openapi_to_tools(doc)
        self.assertRegex(tools[0]["name"], r"^[A-Za-z0-9_.-]{1,64}$")
        self.assertLessEqual(len(tools[0]["description"]), 1000)


if __name__ == "__main__":
    unittest.main()
