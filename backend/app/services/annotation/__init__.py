"""Annotation sub-services (Story 1.6 + queued 1.4).

The ``annotation_service.py`` module at the parent level holds the
core CRUD / assign / submit primitives. This package is for the
extensions that sit on top:

- ``promotion``       — Story 1.6: materialize labeled rows into the
                        project's synthetic / alignment training data.
- ``active_learning`` — queued Story 1.4: uncertainty-ranked next-row
                        selection + inter-annotator agreement.
"""
