"""Data ingestion layer — loads and validates the Excel workbook."""
from .loader import load_workbook, get_workbook_summary

__all__ = ["load_workbook", "get_workbook_summary"]
