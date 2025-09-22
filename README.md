# Playgrround - Table Structure Analysis Library

## Project Goal

This codebase implements a comprehensive table structure analysis system that automatically identifies and classifies different types of rows in tabular data (spreadsheets, CSVs, etc.). The goal is to distinguish between:

- **Header rows** (column labels, titles)
- **Data rows** (actual content records)
- **Total/summary rows** (aggregated calculations)
- **Blank/separator rows** (visual spacing)

## Approach

We extract multiple types of fingerprint metrics from each row:

1. **Value-based metrics (M1-M4)** - Cell content analysis (data types, patterns, position)
2. **Excel style metrics (B1-B3)** - Formatting analysis (bold, borders, fills, alignment)
3. **Continuity metrics (C1)** - Multi-row pattern analysis (how rows relate to surrounding context)

These metrics feed into machine learning models that can automatically parse complex tables and extract structured data, enabling better data processing pipelines for financial reports, invoices, and other semi-structured documents.

---

# Value-Based Metrics

A Python library for extracting row-level value-based fingerprint metrics from DataFrames. This library analyzes tabular data structure, content types, and patterns to identify special rows like headers, totals, and data rows.