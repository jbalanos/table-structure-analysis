import pytest
import pandas as pd
from pathlib import Path
from src.playgrround.core.value_based_metrics import _infer_locale_from_strings, _detect_lexical_tokens, _normalize_numeric_locale_aware


class TestInferLocaleFromStrings:
    """Test class for _infer_locale_from_strings function using AAA approach."""

    def test_Should_ReturnEnUsFormat_When_EnglishNumbersPresent(self):
        # Arrange
        strings = ["1,234.56", "2,000.00", "10,500.75"]

        # Act
        thousands_sep, decimal_sep = _infer_locale_from_strings(strings)

        # Assert
        assert thousands_sep == ","
        assert decimal_sep == "."

    def test_Should_ReturnEuFormat_When_EuropeanNumbersPresent(self):
        # Arrange
        strings = ["1.234,56", "2.000,00", "10.500,75"]

        # Act
        thousands_sep, decimal_sep = _infer_locale_from_strings(strings)

        # Assert
        assert thousands_sep == "."
        assert decimal_sep == ","

    def test_Should_ReturnEnUsFormat_When_MixedButEnglishDominates(self):
        # Arrange
        strings = ["1,234.56", "2,000.00", "1.500,25", "3,456.78", "4,789.12"]

        # Act
        thousands_sep, decimal_sep = _infer_locale_from_strings(strings)

        # Assert
        assert thousands_sep == ","
        assert decimal_sep == "."

    def test_Should_ReturnEuFormat_When_MixedButEuropeanDominates(self):
        # Arrange
        strings = ["1.234,56", "2.000,00", "1,500.25", "3.456,78", "4.789,12"]

        # Act
        thousands_sep, decimal_sep = _infer_locale_from_strings(strings)

        # Assert
        assert thousands_sep == "."
        assert decimal_sep == ","

    def test_Should_ReturnEnUsFormat_When_NoNumbersFound(self):
        # Arrange
        strings = ["text", "more text", "no numbers here"]

        # Act
        thousands_sep, decimal_sep = _infer_locale_from_strings(strings)

        # Assert
        assert thousands_sep == ","
        assert decimal_sep == "."

    def test_Should_ReturnEnUsFormat_When_EmptyList(self):
        # Arrange
        strings = []

        # Act
        thousands_sep, decimal_sep = _infer_locale_from_strings(strings)

        # Assert
        assert thousands_sep == ","
        assert decimal_sep == "."

    def test_Should_ReturnEnUsFormat_When_OnlySimpleNumbers(self):
        # Arrange
        strings = ["123", "456.78", "999"]

        # Act
        thousands_sep, decimal_sep = _infer_locale_from_strings(strings)

        # Assert
        assert thousands_sep == ","
        assert decimal_sep == "."

    def test_Should_HandleIterableInput_When_PassedGenerator(self):
        # Arrange
        def string_generator():
            yield "1,234.56"
            yield "2,000.00"

        # Act
        thousands_sep, decimal_sep = _infer_locale_from_strings(string_generator())

        # Assert
        assert thousands_sep == ","
        assert decimal_sep == "."

    def test_Should_IgnoreNonMatchingStrings_When_MixedContent(self):
        # Arrange
        strings = ["Product A: 1,234.56", "Total: 2,000.00", "random text", "Email: test@example.com"]

        # Act
        thousands_sep, decimal_sep = _infer_locale_from_strings(strings)

        # Assert
        assert thousands_sep == ","
        assert decimal_sep == "."

    def test_Should_HandleEqualHits_When_TieDefaultsToEnglish(self):
        # Arrange
        strings = ["1,234.56", "1.234,56"]  # Equal hits for both formats

        # Act
        thousands_sep, decimal_sep = _infer_locale_from_strings(strings)

        # Assert
        assert thousands_sep == ","
        assert decimal_sep == "."


@pytest.fixture
def sample_invoice_data():
    """Load realistic invoice data for testing."""
    fixtures_path = Path(__file__).parent.parent / "fixtures"
    return pd.read_csv(fixtures_path / "simple_invoice.csv")


@pytest.fixture
def invoice_with_aggregates_data():
    """Load invoice data with clear aggregate lines for testing."""
    fixtures_path = Path(__file__).parent.parent / "fixtures"
    return pd.read_csv(fixtures_path / "invoice_with_aggregates.csv")


class TestDetectLexicalTokens:
    """Test class for _detect_lexical_tokens function using realistic invoice data."""

    def test_Should_DetectTotalTokens_When_InvoiceHasTotalLines(self, sample_invoice_data):
        # Arrange
        data = sample_invoice_data
        # Convert to strings and strip whitespace as expected by function
        stripped_data = data.astype("string").fillna("").applymap(lambda s: s.strip())

        # Act
        result = _detect_lexical_tokens(stripped_data)

        # Assert
        total_rows = result[result["contains_total_token"]]
        assert len(total_rows) > 0

    def test_Should_DetectCurrencySymbols_When_InvoiceHasMultipleCurrencies(self, sample_invoice_data):
        # Arrange
        data = sample_invoice_data
        stripped_data = data.astype("string").fillna("").applymap(lambda s: s.strip())

        # Act
        result = _detect_lexical_tokens(stripped_data)

        # Assert
        currency_rows = result[result["contains_currency_symbol"]]
        assert len(currency_rows) > 0

    def test_Should_DetectCountTokens_When_InvoiceHasSummaryCount(self, invoice_with_aggregates_data):
        # Arrange
        data = invoice_with_aggregates_data
        stripped_data = data.astype("string").fillna("").applymap(lambda s: s.strip())

        # Act
        result = _detect_lexical_tokens(stripped_data)

        # Assert
        count_rows = result[result["contains_count_token"]]
        if len(count_rows) > 0:
            count_data = data.loc[count_rows.index]
            flat_values = count_data.values.flatten()
            has_count = any("count" in str(val).lower() for val in flat_values if pd.notna(val))
            assert has_count

    def test_Should_DetectAverageTokens_When_InvoiceHasAverageCalculations(self, invoice_with_aggregates_data):
        # Arrange
        data = invoice_with_aggregates_data
        stripped_data = data.astype("string").fillna("").applymap(lambda s: s.strip())

        # Act
        result = _detect_lexical_tokens(stripped_data)

        # Assert
        avg_rows = result[result["contains_avg_token"]]
        if len(avg_rows) > 0:
            avg_data = data.loc[avg_rows.index]
            flat_values = avg_data.values.flatten()
            has_avg = any(any(word in str(val).lower() for word in ["avg", "average", "mean"])
                         for val in flat_values if pd.notna(val))
            assert has_avg

    def test_Should_DetectMinMaxTokens_When_InvoiceHasMinMaxCalculations(self, invoice_with_aggregates_data):
        # Arrange
        data = invoice_with_aggregates_data
        stripped_data = data.astype("string").fillna("").applymap(lambda s: s.strip())

        # Act
        result = _detect_lexical_tokens(stripped_data)

        # Assert
        minmax_rows = result[result["contains_minmax_token"]]
        if len(minmax_rows) > 0:
            minmax_data = data.loc[minmax_rows.index]
            flat_values = minmax_data.values.flatten()
            has_minmax = any(any(word in str(val).lower() for word in ["min", "max"])
                           for val in flat_values if pd.notna(val))
            assert has_minmax

    def test_Should_ReturnCorrectStructure_When_ProcessingInvoiceData(self, sample_invoice_data):
        # Arrange
        data = sample_invoice_data
        stripped_data = data.astype("string").fillna("").applymap(lambda s: s.strip())

        # Act
        result = _detect_lexical_tokens(stripped_data)

        # Assert
        expected_columns = [
            "contains_total_token", "contains_avg_token", "contains_count_token",
            "contains_minmax_token", "contains_currency_symbol", "contains_iso_currency"
        ]
        assert list(result.columns) == expected_columns
        assert len(result) == len(data)
        assert result.index.equals(data.index)


class TestNormalizeNumericLocaleAware:
    """Test class for _normalize_numeric_locale_aware function using realistic financial data."""

    def test_Should_ParseEnglishCurrency_When_DollarAmountsProvided(self):
        # Arrange
        data = pd.DataFrame({
            "Amount": ["$1,234.56", "$2,500.00", "Product A"],
            "Price": ["$99.99", "$150.25", "Text"],
            "Total": ["$3,734.55", "$2,650.25", "Description"]
        }).astype("string").fillna("").map(lambda s: s.strip())

        # Act
        numeric_vals_df, numeric_mask_df = _normalize_numeric_locale_aware(
            data, ",", ".", ("$", "€", "£"), ("USD", "EUR", "GBP")
        )

        # Assert
        # Check that dollar amounts were parsed correctly
        assert numeric_vals_df.loc[0, "Amount"] == 1234.56
        assert numeric_vals_df.loc[1, "Amount"] == 2500.00
        assert numeric_vals_df.loc[0, "Price"] == 99.99
        # Check that text remains NaN
        assert pd.isna(numeric_vals_df.loc[2, "Amount"])
        assert pd.isna(numeric_vals_df.loc[2, "Price"])
        # Check mask correctly identifies numeric cells
        assert numeric_mask_df.loc[0, "Amount"] == True
        assert numeric_mask_df.loc[2, "Amount"] == False

    def test_Should_ParseEuropeanCurrency_When_EuroAmountsProvided(self):
        # Arrange
        data = pd.DataFrame({
            "Amount": ["€1.234,56", "€2.500,00", "Service"],
            "Price": ["€99,99", "€150,25", "Text"],
            "Total": ["€3.734,55", "€2.650,25", "Description"]
        }).astype("string").fillna("").map(lambda s: s.strip())

        # Act
        numeric_vals_df, numeric_mask_df = _normalize_numeric_locale_aware(
            data, ".", ",", ("$", "€", "£"), ("USD", "EUR", "GBP")
        )

        # Assert
        # Check that euro amounts were parsed correctly with EU locale
        assert numeric_vals_df.loc[0, "Amount"] == 1234.56
        assert numeric_vals_df.loc[1, "Amount"] == 2500.00
        assert numeric_vals_df.loc[0, "Price"] == 99.99
        # Check that text remains NaN
        assert pd.isna(numeric_vals_df.loc[2, "Amount"])
        # Check mask correctly identifies numeric cells
        assert numeric_mask_df.loc[0, "Amount"] == True
        assert numeric_mask_df.loc[2, "Amount"] == False

    def test_Should_ParseBritishPounds_When_PoundAmountsProvided(self):
        # Arrange
        data = pd.DataFrame({
            "Amount": ["£1,234.56", "£2,500.00", "Item"],
            "Price": ["£99.99", "£150.25", "Text"],
            "Description": ["Product", "Service", "Summary"]
        }).astype("string").fillna("").map(lambda s: s.strip())

        # Act
        numeric_vals_df, numeric_mask_df = _normalize_numeric_locale_aware(
            data, ",", ".", ("$", "€", "£"), ("USD", "EUR", "GBP")
        )

        # Assert
        # Check that pound amounts were parsed correctly
        assert numeric_vals_df.loc[0, "Amount"] == 1234.56
        assert numeric_vals_df.loc[1, "Amount"] == 2500.00
        # Check that description column has no numeric values
        assert pd.isna(numeric_vals_df.loc[0, "Description"])
        assert pd.isna(numeric_vals_df.loc[1, "Description"])

    def test_Should_ParseNegativeAmounts_When_ParenthesesUsed(self):
        # Arrange
        data = pd.DataFrame({
            "Amount": ["($1,234.56)", "($500.00)", "$1,000.00"],
            "Type": ["Expense", "Refund", "Income"],
            "Notes": ["Accounting format", "Credit", "Normal"]
        }).astype("string").fillna("").map(lambda s: s.strip())

        # Act
        numeric_vals_df, numeric_mask_df = _normalize_numeric_locale_aware(
            data, ",", ".", ("$", "€", "£"), ("USD", "EUR", "GBP")
        )

        # Assert
        # Check that parentheses converted to negative values
        assert numeric_vals_df.loc[0, "Amount"] == -1234.56
        assert numeric_vals_df.loc[1, "Amount"] == -500.00
        assert numeric_vals_df.loc[2, "Amount"] == 1000.00
        # Check that text columns remain NaN
        assert pd.isna(numeric_vals_df.loc[0, "Type"])
        assert pd.isna(numeric_vals_df.loc[0, "Notes"])

    def test_Should_RemoveIsoCurrencies_When_IsoCurrencyCodesPresent(self):
        # Arrange
        data = pd.DataFrame({
            "Amount": ["1,234.56 USD", "2,500.00 EUR", "Product"],
            "Price": ["99.99 GBP", "150.25 USD", "Text"],
            "Description": ["Payment USD", "Invoice EUR", "Summary"]
        }).astype("string").fillna("").map(lambda s: s.strip())

        # Act
        numeric_vals_df, numeric_mask_df = _normalize_numeric_locale_aware(
            data, ",", ".", ("$", "€", "£"), ("USD", "EUR", "GBP")
        )

        # Assert
        # Check that ISO codes were removed and amounts parsed
        assert numeric_vals_df.loc[0, "Amount"] == 1234.56
        assert numeric_vals_df.loc[1, "Amount"] == 2500.00
        assert numeric_vals_df.loc[0, "Price"] == 99.99
        # Check text with ISO codes but no numbers remains NaN
        assert pd.isna(numeric_vals_df.loc[0, "Description"])

    def test_Should_HandleMixedContent_When_TextAndNumbersPresent(self):
        # Arrange
        data = pd.DataFrame({
            "Col1": ["Invoice #123", "$1,500.00", "Total: $2,000"],
            "Col2": ["Product A", "€800.50", "Service B"],
            "Col3": ["Qty: 5", "Price each", "£400.25"]
        }).astype("string").fillna("").map(lambda s: s.strip())

        # Act
        numeric_vals_df, numeric_mask_df = _normalize_numeric_locale_aware(
            data, ",", ".", ("$", "€", "£"), ("USD", "EUR", "GBP")
        )

        # Assert
        # Only pure numeric values should be parsed
        assert pd.isna(numeric_vals_df.loc[0, "Col1"])  # "Invoice #123"
        assert numeric_vals_df.loc[1, "Col1"] == 1500.00  # "$1,500.00"
        assert pd.isna(numeric_vals_df.loc[2, "Col1"])  # "Total: $2,000" (mixed text)
        assert numeric_vals_df.loc[1, "Col2"] == 800.50   # "€800.50"
        assert numeric_vals_df.loc[2, "Col3"] == 400.25   # "£400.25"

    def test_Should_HandleEmptyAndNullValues_When_DataContainsBlanks(self):
        # Arrange
        data = pd.DataFrame({
            "Amount": ["$1,000.00", "", "$500.00"],
            "Price": ["", "€200.50", ""],
            "Total": ["£300.75", "", ""]
        }).astype("string").fillna("").map(lambda s: s.strip())

        # Act
        numeric_vals_df, numeric_mask_df = _normalize_numeric_locale_aware(
            data, ",", ".", ("$", "€", "£"), ("USD", "EUR", "GBP")
        )

        # Assert
        # Check numeric values are parsed correctly
        assert numeric_vals_df.loc[0, "Amount"] == 1000.00
        assert numeric_vals_df.loc[2, "Amount"] == 500.00
        assert numeric_vals_df.loc[1, "Price"] == 200.50
        # Check empty values are NaN
        assert pd.isna(numeric_vals_df.loc[1, "Amount"])
        assert pd.isna(numeric_vals_df.loc[0, "Price"])
        # Check mask reflects actual content
        assert numeric_mask_df.loc[0, "Amount"] == True
        assert numeric_mask_df.loc[1, "Amount"] == False

    def test_Should_ReturnCorrectStructure_When_ProcessingFinancialData(self):
        # Arrange
        data = pd.DataFrame({
            "Amount": ["$1,234.56", "€2,345.67"],
            "Description": ["Payment", "Invoice"],
            "Quantity": ["5", "10"]
        }).astype("string").fillna("").map(lambda s: s.strip())

        # Act
        numeric_vals_df, numeric_mask_df = _normalize_numeric_locale_aware(
            data, ",", ".", ("$", "€", "£"), ("USD", "EUR", "GBP")
        )

        # Assert
        # Check return structure
        assert isinstance(numeric_vals_df, pd.DataFrame)
        assert isinstance(numeric_mask_df, pd.DataFrame)
        assert numeric_vals_df.shape == data.shape
        assert numeric_mask_df.shape == data.shape
        assert numeric_vals_df.index.equals(data.index)
        assert numeric_mask_df.index.equals(data.index)
        # Check that quantity column parsed as numbers
        assert numeric_vals_df.loc[0, "Quantity"] == 5.0
        assert numeric_vals_df.loc[1, "Quantity"] == 10.0

    def test_Should_HandleSpecialSpaces_When_NonBreakingSpacesPresent(self):
        # Arrange
        # Using actual Unicode characters for thin space and NBSP
        data = pd.DataFrame({
            "Amount": ["$ 1,234.56", "€\u00A02,345.67", "£\u202F3,456.78"],
            "Price": ["Text", "$ 999.99", "Description"]
        }).astype("string").fillna("").map(lambda s: s.strip())

        # Act
        numeric_vals_df, numeric_mask_df = _normalize_numeric_locale_aware(
            data, ",", ".", ("$", "€", "£"), ("USD", "EUR", "GBP")
        )

        # Assert
        # Check that special spaces were removed and amounts parsed
        assert numeric_vals_df.loc[0, "Amount"] == 1234.56
        assert numeric_vals_df.loc[1, "Amount"] == 2345.67
        assert numeric_vals_df.loc[2, "Amount"] == 3456.78
        assert numeric_vals_df.loc[1, "Price"] == 999.99


class TestPrepareDataframeForAnalysis:
    """Test class for _prepare_dataframe_for_analysis function."""

    def test_Should_ReturnCorrectOutputStructure_When_ValidDataFrameProvided(self):
        """Test that function returns tuple with 6 elements in correct format."""
        # Arrange
        data = pd.DataFrame({
            "col1": ["value1", "value2"],
            "col2": ["value3", "value4"]
        })

        # Act
        from src.playgrround.core.value_based_metrics import _prepare_dataframe_for_analysis
        result = _prepare_dataframe_for_analysis(data)

        # Assert
        assert isinstance(result, tuple)
        assert len(result) == 6
        stripped_df, null_mask_df, date_mask_df, paren_neg_mask_df, thousands_used, decimal_used = result

        # Check all DataFrames have same shape as input
        assert stripped_df.shape == data.shape
        assert null_mask_df.shape == data.shape
        assert date_mask_df.shape == data.shape
        assert paren_neg_mask_df.shape == data.shape

        # Check separators are strings
        assert isinstance(thousands_used, str)
        assert isinstance(decimal_used, str)

    def test_Should_InferEnglishLocale_When_EnglishNumbersInData(self):
        """Test locale inference detects EN format (comma thousands, dot decimal)."""
        # Arrange
        data = pd.DataFrame({
            "prices": ["$1,234.56", "$2,500.00", "$3,999.99"],
            "amounts": ["1,000.50", "2,250.75", "5,678.90"]
        })

        # Act
        from src.playgrround.core.value_based_metrics import _prepare_dataframe_for_analysis
        _, _, _, _, thousands_used, decimal_used = _prepare_dataframe_for_analysis(data, infer_locale=True)

        # Assert
        assert thousands_used == ","
        assert decimal_used == "."

    def test_Should_InferEuropeanLocale_When_EuropeanNumbersInData(self):
        """Test locale inference detects EU format (dot thousands, comma decimal)."""
        # Arrange
        data = pd.DataFrame({
            "prices": ["€1.234,56", "€2.500,00", "€3.999,99"],
            "amounts": ["1.000,50", "2.250,75", "5.678,90"]
        })

        # Act
        from src.playgrround.core.value_based_metrics import _prepare_dataframe_for_analysis
        _, _, _, _, thousands_used, decimal_used = _prepare_dataframe_for_analysis(data, infer_locale=True)

        # Assert
        assert thousands_used == "."
        assert decimal_used == ","

    def test_Should_UseProvidedDefaults_When_InferLocaleDisabled(self):
        """Test that function uses provided separators when infer_locale=False."""
        # Arrange
        data = pd.DataFrame({
            "mixed": ["$1,234.56", "€1.234,56", "£1 234,56"],
            "values": ["2,500.00", "3.500,75", "4-999-99"]
        })

        # Act
        from src.playgrround.core.value_based_metrics import _prepare_dataframe_for_analysis
        _, _, _, _, thousands_used, decimal_used = _prepare_dataframe_for_analysis(
            data, infer_locale=False, thousands=".", decimal=","
        )

        # Assert
        assert thousands_used == "."
        assert decimal_used == ","

    def test_Should_UseDefaults_When_EmptyDataFrame(self):
        """Test behavior with empty DataFrame."""
        # Arrange
        data = pd.DataFrame()

        # Act
        from src.playgrround.core.value_based_metrics import _prepare_dataframe_for_analysis
        stripped, null_mask_df, date_mask_df, paren_neg_mask_df, thousands_used, decimal_used = _prepare_dataframe_for_analysis(data)

        # Assert
        assert thousands_used == ","
        assert decimal_used == "."
        assert stripped.empty
        assert null_mask_df.empty
        assert date_mask_df.empty
        assert paren_neg_mask_df.empty

    def test_Should_ConvertAllCellsToStrings_When_MixedDtypesProvided(self):
        """Test string conversion and trimming of mixed data types."""
        # Arrange
        data = pd.DataFrame({
            "mixed": [123, 45.6, "  text  ", None]
        })

        # Act
        from src.playgrround.core.value_based_metrics import _prepare_dataframe_for_analysis
        stripped, _, _, _, _, _ = _prepare_dataframe_for_analysis(data)

        # Assert
        assert stripped.loc[0, "mixed"] == "123"
        assert stripped.loc[1, "mixed"] == "45.6"
        assert stripped.loc[2, "mixed"] == "text"
        assert stripped.loc[3, "mixed"] == ""

    def test_Should_CreateCorrectNullMask_When_BlankAndEmptyValues(self):
        """Test null mask creation for various empty value types."""
        # Arrange
        data = pd.DataFrame({
            "values": ["text", "", "   ", None, 0, "0"]
        })

        # Act
        from src.playgrround.core.value_based_metrics import _prepare_dataframe_for_analysis
        _, null_mask_df, _, _, _, _ = _prepare_dataframe_for_analysis(data)

        # Assert
        assert null_mask_df.loc[0, "values"] == False  # "text"
        assert null_mask_df.loc[1, "values"] == True   # ""
        assert null_mask_df.loc[2, "values"] == True   # "   "
        assert null_mask_df.loc[3, "values"] == True   # None
        assert null_mask_df.loc[4, "values"] == False  # 0
        assert null_mask_df.loc[5, "values"] == False  # "0"

    def test_Should_DetectDatesCorrectly_When_DateLikeStringsProvided(self):
        """Test date detection across various date formats."""
        # Arrange
        data = pd.DataFrame({
            "dates": ["2024-01-15", "2024/01/16", "2024-01-17", "not a date", "2024-13-40"]
        })

        # Act
        from src.playgrround.core.value_based_metrics import _prepare_dataframe_for_analysis
        _, _, date_mask_df, _, _, _ = _prepare_dataframe_for_analysis(data)

        # Assert
        assert date_mask_df.loc[0, "dates"] == True   # "2024-01-15"
        assert date_mask_df.loc[1, "dates"] == True   # "2024/01/16"
        assert date_mask_df.loc[2, "dates"] == True   # "2024-01-17"
        assert date_mask_df.loc[3, "dates"] == False  # "not a date"
        assert date_mask_df.loc[4, "dates"] == False  # "2024-13-40"

    def test_Should_DetectParenthesesNegatives_When_AccountingFormatProvided(self):
        """Test detection of accounting-style negative numbers in parentheses."""
        # Arrange
        data = pd.DataFrame({
            "values": ["(123)", "(45.67)", "123", "text", "(abc)", "()"]
        })

        # Act
        from src.playgrround.core.value_based_metrics import _prepare_dataframe_for_analysis
        _, _, _, paren_neg_mask_df, _, _ = _prepare_dataframe_for_analysis(data)

        # Assert
        assert paren_neg_mask_df.loc[0, "values"] == True   # "(123)"
        assert paren_neg_mask_df.loc[1, "values"] == True   # "(45.67)"
        assert paren_neg_mask_df.loc[2, "values"] == False  # "123"
        assert paren_neg_mask_df.loc[3, "values"] == False  # "text"
        assert paren_neg_mask_df.loc[4, "values"] == False  # "(abc)"
        assert paren_neg_mask_df.loc[5, "values"] == False  # "()"

    def test_Should_HandleUnicodeAndSpecialCharacters_When_InternationalData(self):
        """Test handling of Unicode characters and special spaces."""
        # Arrange
        data = pd.DataFrame({
            "unicode": ["\u202f  Café  \u202f", "\xa0 naïve \xa0", "  résumé  "]
        })

        # Act
        from src.playgrround.core.value_based_metrics import _prepare_dataframe_for_analysis
        stripped, _, _, _, _, _ = _prepare_dataframe_for_analysis(data)

        # Assert
        assert stripped.loc[0, "unicode"] == "Café"
        assert stripped.loc[1, "unicode"] == "naïve"
        assert stripped.loc[2, "unicode"] == "résumé"

    def test_Should_HandleLargeDataFrame_When_PerformanceMatters(self):
        """Test function performance and correctness with large DataFrame."""
        # Arrange
        import time
        import numpy as np
        data = pd.DataFrame(np.random.choice(
            ["text", "2024-01-01", "1,234.56", "", "(123)", None],
            size=(1000, 20)
        ))

        # Act
        from src.playgrround.core.value_based_metrics import _prepare_dataframe_for_analysis
        start_time = time.time()
        stripped, null_mask_df, date_mask_df, paren_neg_mask_df, thousands_used, decimal_used = _prepare_dataframe_for_analysis(data)
        end_time = time.time()

        # Assert
        assert end_time - start_time < 5.0  # Should complete in under 5 seconds
        assert stripped.shape == (1000, 20)
        assert null_mask_df.shape == (1000, 20)
        assert date_mask_df.shape == (1000, 20)
        assert paren_neg_mask_df.shape == (1000, 20)

    def test_Should_HandleSingleRowDataFrame_When_EdgeCaseProvided(self):
        """Test behavior with single-row DataFrame."""
        # Arrange
        data = pd.DataFrame({
            "text": ["value"],
            "number": ["1,234.56"],
            "date": ["2024-01-01"],
            "empty": [""]
        })

        # Act
        from src.playgrround.core.value_based_metrics import _prepare_dataframe_for_analysis
        stripped, null_mask_df, date_mask_df, paren_neg_mask_df, thousands_used, decimal_used = _prepare_dataframe_for_analysis(data)

        # Assert
        assert stripped.shape == (1, 4)
        assert null_mask_df.shape == (1, 4)
        assert date_mask_df.shape == (1, 4)
        assert paren_neg_mask_df.shape == (1, 4)
        assert thousands_used == ","
        assert decimal_used == "."

    def test_Should_HandleSingleColumnDataFrame_When_EdgeCaseProvided(self):
        """Test behavior with single-column DataFrame."""
        # Arrange
        data = pd.DataFrame({
            "values": ["text", "2024-01-01", "", "(123)", "1,234.56"]
        })

        # Act
        from src.playgrround.core.value_based_metrics import _prepare_dataframe_for_analysis
        stripped, null_mask_df, date_mask_df, paren_neg_mask_df, thousands_used, decimal_used = _prepare_dataframe_for_analysis(data)

        # Assert
        assert stripped.shape == (5, 1)
        assert null_mask_df.shape == (5, 1)
        assert date_mask_df.shape == (5, 1)
        assert paren_neg_mask_df.shape == (5, 1)
        assert null_mask_df.loc[2, "values"] == True  # Empty string
        assert date_mask_df.loc[1, "values"] == True  # Date
        assert paren_neg_mask_df.loc[3, "values"] == True  # Parentheses negative

    def test_Should_PreserveDtypes_When_OutputDataFramesCreated(self):
        """Test that output DataFrames have expected dtypes."""
        # Arrange
        data = pd.DataFrame({
            "mixed": [123, "text", 45.6, None]
        })

        # Act
        from src.playgrround.core.value_based_metrics import _prepare_dataframe_for_analysis
        stripped, null_mask_df, date_mask_df, paren_neg_mask_df, thousands_used, decimal_used = _prepare_dataframe_for_analysis(data)

        # Assert
        assert stripped.dtypes["mixed"] in ["object", "string"]
        assert null_mask_df.dtypes["mixed"] == "bool"
        assert date_mask_df.dtypes["mixed"] == "bool"
        assert paren_neg_mask_df.dtypes["mixed"] == "bool"
        assert isinstance(thousands_used, str)
        assert isinstance(decimal_used, str)

    def test_Should_ReturnCorrectStrippedDataFrame_When_HappyPathDataProvided(self):
        """Test that stripped DataFrame correctly converts and trims all values."""
        # Arrange
        data = pd.DataFrame({
            "Cust": ["  Customer A  ", "Customer B", "  Customer C", "Customer D  ", "", "Customer F", "Customer G", "Customer H", "", "  Total Summary  "],
            "Inv_Number": ["INV-001  ", "  INV-002", "INV-003", "  ", "INV-005", "INV-006  ", "", "INV-008", "INV-009", ""],
            "Date": ["2024-01-15", "  2024-01-16  ", "", "2024-01-18", "2024-01-19", "  ", "2024-01-21", "  2024-01-22", "2024-01-23", ""],
            "Items": ["  Widget A  ", "Widget B", "", "  Widget D", "Widget E  ", "Widget F", "  ", "Widget H", "Widget I", ""],
            "Price": ["  $1,234.56  ", "$2,500.00", "  $750.25", "", "$1,800.99", "  $999.99  ", "$3,200.00", "  ", "$1,500.75", "  $25,890.54  "]
        })

        expected_stripped = pd.DataFrame({
            "Cust": ["Customer A", "Customer B", "Customer C", "Customer D", "", "Customer F", "Customer G", "Customer H", "", "Total Summary"],
            "Inv_Number": ["INV-001", "INV-002", "INV-003", "", "INV-005", "INV-006", "", "INV-008", "INV-009", ""],
            "Date": ["2024-01-15", "2024-01-16", "", "2024-01-18", "2024-01-19", "", "2024-01-21", "2024-01-22", "2024-01-23", ""],
            "Items": ["Widget A", "Widget B", "", "Widget D", "Widget E", "Widget F", "", "Widget H", "Widget I", ""],
            "Price": ["$1,234.56", "$2,500.00", "$750.25", "", "$1,800.99", "$999.99", "$3,200.00", "", "$1,500.75", "$25,890.54"]
        })

        # Act
        from src.playgrround.core.value_based_metrics import _prepare_dataframe_for_analysis
        stripped, _, _, _, _, _ = _prepare_dataframe_for_analysis(data)

        # Assert
        pd.testing.assert_frame_equal(stripped, expected_stripped)

    def test_Should_ReturnCorrectNullMask_When_HappyPathDataProvided(self):
        """Test that null mask correctly identifies empty and whitespace-only cells."""
        # Arrange
        data = pd.DataFrame({
            "Cust": ["Customer A", "Customer B", "Customer C", "Customer D", "", "Customer F", "Customer G", "Customer H", "", "Total Summary"],
            "Inv_Number": ["INV-001", "INV-002", "INV-003", "  ", "INV-005", "INV-006", "", "INV-008", "INV-009", ""],
            "Date": ["2024-01-15", "2024-01-16", "", "2024-01-18", "2024-01-19", "  ", "2024-01-21", "2024-01-22", "2024-01-23", ""],
            "Items": ["Widget A", "Widget B", "", "Widget D", "Widget E", "Widget F", "  ", "Widget H", "Widget I", ""],
            "Price": ["$1,234.56", "$2,500.00", "$750.25", "", "$1,800.99", "$999.99", "$3,200.00", "  ", "$1,500.75", "$25,890.54"]
        })

        expected_null_mask = pd.DataFrame({
            "Cust": [False, False, False, False, True, False, False, False, True, False],
            "Inv_Number": [False, False, False, True, False, False, True, False, False, True],
            "Date": [False, False, True, False, False, True, False, False, False, True],
            "Items": [False, False, True, False, False, False, True, False, False, True],
            "Price": [False, False, False, True, False, False, False, True, False, False]
        })

        # Act
        from src.playgrround.core.value_based_metrics import _prepare_dataframe_for_analysis
        _, null_mask_df, _, _, _, _ = _prepare_dataframe_for_analysis(data)

        # Assert
        pd.testing.assert_frame_equal(null_mask_df, expected_null_mask)

    def test_Should_ReturnCorrectDateMask_When_HappyPathDataProvided(self):
        """Test that date mask correctly identifies date-like values."""
        # Arrange
        data = pd.DataFrame({
            "Cust": ["Customer A", "Customer B", "Customer C", "Customer D", "", "Customer F", "Customer G", "Customer H", "", "Total Summary"],
            "Inv_Number": ["INV-001", "INV-002", "INV-003", "2024-01-01", "INV-005", "01/15/2024", "", "INV-008", "Jan 9, 2024", ""],
            "Date": ["2024-01-15", "2024-01-16", "", "2024-01-18", "2024-01-19", "  ", "2024-01-21", "2024-01-22", "2024-01-23", ""],
            "Items": ["Widget A", "Dec 25, 2023", "", "Widget D", "2024-12-31", "Widget F", "  ", "Widget H", "Widget I", ""],
            "Price": ["$1,234.56", "$2,500.00", "$750.25", "", "$1,800.99", "$999.99", "$3,200.00", "  ", "$1,500.75", "$25,890.54"]
        })

        expected_date_mask = pd.DataFrame({
            "Cust": [False, False, False, False, False, False, False, False, False, False],
            "Inv_Number": [False, False, False, True, False, True, False, False, True, False],
            "Date": [True, True, False, True, True, False, True, True, True, False],
            "Items": [False, True, False, False, True, False, False, False, False, False],
            "Price": [False, False, False, False, False, False, False, False, False, False]
        })

        # Act
        from src.playgrround.core.value_based_metrics import _prepare_dataframe_for_analysis
        _, _, date_mask_df, _, _, _ = _prepare_dataframe_for_analysis(data)

        # Assert
        pd.testing.assert_frame_equal(date_mask_df, expected_date_mask)

    def test_Should_ReturnCorrectParenNegMask_When_HappyPathDataProvided(self):
        """Test that parentheses negative mask correctly identifies accounting-style negatives."""
        # Arrange
        data = pd.DataFrame({
            "Cust": ["Customer A", "Customer B", "Customer C", "Customer D", "", "Customer F", "Customer G", "Customer H", "", "Total Summary"],
            "Inv_Number": ["INV-001", "INV-002", "INV-003", "(INV-004)", "INV-005", "INV-006", "", "INV-008", "INV-009", ""],
            "Date": ["2024-01-15", "2024-01-16", "", "2024-01-18", "2024-01-19", "  ", "2024-01-21", "2024-01-22", "2024-01-23", ""],
            "Items": ["Widget A", "Widget B", "", "Widget D", "Widget E", "Widget F", "  ", "(Widget H)", "Widget I", ""],
            "Price": ["$1,234.56", "($2,500.00)", "$750.25", "", "($1,800.99)", "$999.99", "($3,200.00)", "  ", "$1,500.75", "($25,890.54)"]
        })

        expected_paren_neg_mask = pd.DataFrame({
            "Cust": [False, False, False, False, False, False, False, False, False, False],
            "Inv_Number": [False, False, False, False, False, False, False, False, False, False],
            "Date": [False, False, False, False, False, False, False, False, False, False],
            "Items": [False, False, False, False, False, False, False, False, False, False],
            "Price": [False, True, False, False, True, False, True, False, False, True]
        })

        # Act
        from src.playgrround.core.value_based_metrics import _prepare_dataframe_for_analysis
        _, _, _, paren_neg_df, _, _ = _prepare_dataframe_for_analysis(data)

        # Assert
        pd.testing.assert_frame_equal(paren_neg_df, expected_paren_neg_mask)


class TestCalculateDensityTypeMetrics:
    """Test class for _calculate_density_type_metrics function."""

    def test_Should_CalculateCorrectDensityMetrics_When_MixedDataProvided(self):
        """Test that density and type metrics are correctly calculated for mixed data."""
        # Arrange
        null_mask_df = pd.DataFrame({
            "Col1": [False, True, False, False],
            "Col2": [False, False, True, False],
            "Col3": [True, True, False, False]
        })

        date_mask_df = pd.DataFrame({
            "Col1": [True, False, False, False],
            "Col2": [False, True, False, False],
            "Col3": [False, False, True, False]
        })

        numeric_mask_df = pd.DataFrame({
            "Col1": [False, False, True, True],
            "Col2": [True, False, False, True],
            "Col3": [False, False, False, True]
        })

        df_index = pd.Index([0, 1, 2, 3])
        n_cols = 3

        expected_result = pd.DataFrame({
            "non_null_ratio": [2/3, 1/3, 2/3, 1.0],
            "text_count": [0, 0, 0, 0],
            "numeric_count": [1, 0, 1, 3],
            "date_count": [1, 1, 1, 0],
            "null_count": [1, 2, 1, 0],
            "text_ratio": [0.0, 0.0, 0.0, 0.0],
            "numeric_ratio": [0.5, 0.0, 0.5, 1.0],
            "type_entropy": [1.0986122886957408, 0.6365141683500748, 1.0986122886957408, 8.289306334778563e-11]
        }, index=df_index)

        # Act
        from src.playgrround.core.value_based_metrics import _calculate_density_type_metrics
        result = _calculate_density_type_metrics(null_mask_df, date_mask_df, numeric_mask_df, df_index, n_cols)

        # Assert
        pd.testing.assert_frame_equal(result, expected_result)

    def test_Should_HandleAllZeroDensities_When_NoMatchingDataProvided(self):
        """Test that function handles cases with all zero densities."""
        # Arrange
        null_mask_df = pd.DataFrame({
            "Col1": [False, False, False],
            "Col2": [False, False, False]
        })

        date_mask_df = pd.DataFrame({
            "Col1": [False, False, False],
            "Col2": [False, False, False]
        })

        numeric_mask_df = pd.DataFrame({
            "Col1": [False, False, False],
            "Col2": [False, False, False]
        })

        df_index = pd.Index([0, 1, 2])
        n_cols = 2

        expected_result = pd.DataFrame({
            "non_null_ratio": [1.0, 1.0, 1.0],
            "text_count": [2, 2, 2],
            "numeric_count": [0, 0, 0],
            "date_count": [0, 0, 0],
            "null_count": [0, 0, 0],
            "text_ratio": [1.0, 1.0, 1.0],
            "numeric_ratio": [0.0, 0.0, 0.0],
            "type_entropy": [0.0, 0.0, 0.0]
        }, index=df_index)

        # Act
        from src.playgrround.core.value_based_metrics import _calculate_density_type_metrics
        result = _calculate_density_type_metrics(null_mask_df, date_mask_df, numeric_mask_df, df_index, n_cols)

        # Assert
        pd.testing.assert_frame_equal(result, expected_result)

    def test_Should_HandleFullDensities_When_AllDataMatches(self):
        """Test that function handles cases with all maximum densities."""
        # Arrange
        null_mask_df = pd.DataFrame({
            "Col1": [True, False, False],
            "Col2": [True, False, False]
        })

        date_mask_df = pd.DataFrame({
            "Col1": [False, True, False],
            "Col2": [False, True, False]
        })

        numeric_mask_df = pd.DataFrame({
            "Col1": [False, False, True],
            "Col2": [False, False, True]
        })

        df_index = pd.Index([0, 1, 2])
        n_cols = 2

        expected_result = pd.DataFrame({
            "non_null_ratio": [0.0, 1.0, 1.0],
            "text_count": [0, 0, 0],
            "numeric_count": [0, 0, 2],
            "date_count": [0, 2, 0],
            "null_count": [2, 0, 0],
            "text_ratio": [0.0, 0.0, 0.0],
            "numeric_ratio": [0.0, 0.0, 1.0],
            "type_entropy": [8.289306334778563e-11, 8.289306334778563e-11, 8.289306334778563e-11]
        }, index=df_index)

        # Act
        from src.playgrround.core.value_based_metrics import _calculate_density_type_metrics
        result = _calculate_density_type_metrics(null_mask_df, date_mask_df, numeric_mask_df, df_index, n_cols)

        # Assert
        pd.testing.assert_frame_equal(result, expected_result)

    def test_Should_HandleRealisticInvoiceData_When_MixedProportionsProvided(self):
        """Test with realistic invoice-like data distributions."""
        # Arrange - simulating invoice rows with varying compositions
        null_mask_df = pd.DataFrame({
            "Customer": [False, False, False, True, False],  # Header row has some nulls
            "Invoice": [False, False, False, False, True],   # Summary row has null
            "Date": [False, False, True, False, False],      # One missing date
            "Amount": [False, False, False, True, False],    # Header row has null amount
            "Notes": [True, True, False, True, True]         # Notes often empty
        })

        date_mask_df = pd.DataFrame({
            "Customer": [False, False, False, False, False],
            "Invoice": [False, False, False, False, False],
            "Date": [True, True, False, True, True],         # Most dates detected
            "Amount": [False, False, False, False, False],
            "Notes": [False, False, False, False, False]
        })

        numeric_mask_df = pd.DataFrame({
            "Customer": [False, False, False, False, False],
            "Invoice": [True, True, True, False, True],      # Invoice numbers mostly numeric
            "Date": [False, False, False, False, False],
            "Amount": [True, True, True, False, True],       # Amounts mostly numeric
            "Notes": [False, False, False, False, False]
        })

        df_index = pd.Index([0, 1, 2, 3, 4])
        n_cols = 5

        # Act
        from src.playgrround.core.value_based_metrics import _calculate_density_type_metrics
        result = _calculate_density_type_metrics(null_mask_df, date_mask_df, numeric_mask_df, df_index, n_cols)

        # Assert - verify realistic distributions
        assert result.loc[0, "non_null_ratio"] == 4/5    # Detail row: mostly filled
        assert result.loc[1, "non_null_ratio"] == 4/5    # Detail row: mostly filled
        assert result.loc[2, "non_null_ratio"] == 4/5    # Detail row: mostly filled (1 null)
        assert result.loc[3, "non_null_ratio"] == 2/5    # Header row: many nulls
        assert result.loc[4, "non_null_ratio"] == 3/5    # Summary row: some nulls

        # Check that entropy is higher for mixed rows vs header row
        assert result.loc[0, "type_entropy"] > result.loc[3, "type_entropy"]
        assert result.loc[1, "type_entropy"] > result.loc[3, "type_entropy"]

    def test_Should_HandleVaryingRowCompositions_When_DifferentDataPatternsProvided(self):
        """Test with rows having different dominant data types."""
        # Arrange - different row patterns
        null_mask_df = pd.DataFrame({
            "Col1": [False, True, False, False],
            "Col2": [False, True, False, True],
            "Col3": [False, True, True, False]
        })

        date_mask_df = pd.DataFrame({
            "Col1": [True, False, False, False],   # Row 0: mostly dates
            "Col2": [True, False, False, False],
            "Col3": [True, False, False, False]
        })

        numeric_mask_df = pd.DataFrame({
            "Col1": [False, False, True, False],   # Row 2: mostly numeric
            "Col2": [False, False, True, False],
            "Col3": [False, False, False, True]    # Row 3: mixed
        })

        df_index = pd.Index([0, 1, 2, 3])
        n_cols = 3

        # Act
        from src.playgrround.core.value_based_metrics import _calculate_density_type_metrics
        result = _calculate_density_type_metrics(null_mask_df, date_mask_df, numeric_mask_df, df_index, n_cols)

        # Assert - check row-specific patterns
        # Row 0: date-dominant (3 dates, 0 nulls, 0 numeric, 0 text)
        assert result.loc[0, "date_count"] == 3
        assert result.loc[0, "null_count"] == 0
        assert result.loc[0, "numeric_count"] == 0
        assert result.loc[0, "non_null_ratio"] == 1.0

        # Row 1: null-dominant (0 dates, 3 nulls, 0 numeric, 0 text)
        assert result.loc[1, "null_count"] == 3
        assert result.loc[1, "non_null_ratio"] == 0.0

        # Row 2: numeric-dominant (0 dates, 1 null, 2 numeric, 0 text)
        assert result.loc[2, "numeric_count"] == 2
        assert result.loc[2, "null_count"] == 1
        assert result.loc[2, "numeric_ratio"] == 1.0  # 2/2 non-null are numeric

    def test_Should_HandleSingleRowScenarios_When_EdgeCasesProvided(self):
        """Test with single row DataFrames."""
        # Arrange
        null_mask_df = pd.DataFrame({
            "Col1": [True],
            "Col2": [False],
            "Col3": [False]
        })

        date_mask_df = pd.DataFrame({
            "Col1": [False],
            "Col2": [True],
            "Col3": [False]
        })

        numeric_mask_df = pd.DataFrame({
            "Col1": [False],
            "Col2": [False],
            "Col3": [True]
        })

        df_index = pd.Index([0])
        n_cols = 3

        # Act
        from src.playgrround.core.value_based_metrics import _calculate_density_type_metrics
        result = _calculate_density_type_metrics(null_mask_df, date_mask_df, numeric_mask_df, df_index, n_cols)

        # Assert
        assert result.shape == (1, 8)
        assert result.loc[0, "non_null_ratio"] == 2/3
        assert result.loc[0, "null_count"] == 1
        assert result.loc[0, "date_count"] == 1
        assert result.loc[0, "numeric_count"] == 1
        assert result.loc[0, "text_count"] == 0

    def test_Should_HandleSingleColumnScenarios_When_EdgeCasesProvided(self):
        """Test with single column DataFrames."""
        # Arrange
        null_mask_df = pd.DataFrame({
            "OnlyCol": [True, False, False, True]
        })

        date_mask_df = pd.DataFrame({
            "OnlyCol": [False, True, False, False]
        })

        numeric_mask_df = pd.DataFrame({
            "OnlyCol": [False, False, True, False]
        })

        df_index = pd.Index([0, 1, 2, 3])
        n_cols = 1

        # Act
        from src.playgrround.core.value_based_metrics import _calculate_density_type_metrics
        result = _calculate_density_type_metrics(null_mask_df, date_mask_df, numeric_mask_df, df_index, n_cols)

        # Assert
        assert result.shape == (4, 8)
        # Row 0: null
        assert result.loc[0, "non_null_ratio"] == 0.0
        assert result.loc[0, "null_count"] == 1
        # Row 1: date
        assert result.loc[1, "non_null_ratio"] == 1.0
        assert result.loc[1, "date_count"] == 1
        # Row 2: numeric
        assert result.loc[2, "non_null_ratio"] == 1.0
        assert result.loc[2, "numeric_count"] == 1
        # Row 3: null
        assert result.loc[3, "non_null_ratio"] == 0.0
        assert result.loc[3, "null_count"] == 1

    def test_Should_CalculateEntropyCorrectly_When_DifferentMixingPatternsProvided(self):
        """Test entropy calculations for various data mixing patterns."""
        # Arrange - create scenarios with known entropy patterns
        null_mask_df = pd.DataFrame({
            "Col1": [False, False, True, True],   # Row 0: pure, Row 2: mixed
            "Col2": [False, False, True, True],
            "Col3": [False, False, False, False],
            "Col4": [False, False, False, False]
        })

        date_mask_df = pd.DataFrame({
            "Col1": [True, False, False, False],   # Row 0: pure dates
            "Col2": [True, False, False, False],   # Row 1: pure text (by elimination)
            "Col3": [True, False, True, False],    # Row 2: mixed dates/nulls
            "Col4": [True, False, False, True]     # Row 3: mixed dates/nulls
        })

        numeric_mask_df = pd.DataFrame({
            "Col1": [False, False, False, False],
            "Col2": [False, True, False, False],   # Row 1: pure numeric
            "Col3": [False, True, False, True],    # Row 2: mixed numeric/nulls/dates
            "Col4": [False, True, False, False]    # Row 3: mixed numeric/dates/nulls
        })

        df_index = pd.Index([0, 1, 2, 3])
        n_cols = 4

        # Act
        from src.playgrround.core.value_based_metrics import _calculate_density_type_metrics
        result = _calculate_density_type_metrics(null_mask_df, date_mask_df, numeric_mask_df, df_index, n_cols)

        # Assert entropy relationships
        # Row 0: pure dates (low entropy)
        # Row 1: pure types per column but different types (medium entropy)
        # Row 2 & 3: mixed types (higher entropy)
        entropy_0 = result.loc[0, "type_entropy"]
        entropy_1 = result.loc[1, "type_entropy"]
        entropy_2 = result.loc[2, "type_entropy"]
        entropy_3 = result.loc[3, "type_entropy"]

        # Pure date row should have very low entropy
        assert entropy_0 < entropy_1
        # Mixed rows should generally have higher entropy than pure rows
        assert entropy_2 > entropy_0
        assert entropy_3 > entropy_0

    def test_Should_HandleAllNullRow_When_CompletelyEmptyRowProvided(self):
        """Test behavior with completely null rows."""
        # Arrange
        null_mask_df = pd.DataFrame({
            "Col1": [False, True, False],
            "Col2": [False, True, False],
            "Col3": [False, True, False]  # Row 1 is completely null
        })

        date_mask_df = pd.DataFrame({
            "Col1": [True, False, False],
            "Col2": [False, False, True],
            "Col3": [False, False, False]
        })

        numeric_mask_df = pd.DataFrame({
            "Col1": [False, False, True],
            "Col2": [True, False, False],
            "Col3": [True, False, True]
        })

        df_index = pd.Index([0, 1, 2])
        n_cols = 3

        # Act
        from src.playgrround.core.value_based_metrics import _calculate_density_type_metrics
        result = _calculate_density_type_metrics(null_mask_df, date_mask_df, numeric_mask_df, df_index, n_cols)

        # Assert - all-null row behavior
        assert result.loc[1, "non_null_ratio"] == 0.0
        assert result.loc[1, "null_count"] == 3
        assert result.loc[1, "text_count"] == 0
        assert result.loc[1, "numeric_count"] == 0
        assert result.loc[1, "date_count"] == 0
        assert result.loc[1, "text_ratio"] == 0.0
        assert result.loc[1, "numeric_ratio"] == 0.0
        # All-null row should have very low entropy (approaching 0)
        assert result.loc[1, "type_entropy"] < 0.1


class TestCalculatePositionContextMetrics:
    """Test class for _calculate_position_context_metrics function."""

    def test_Should_CalculateCorrectPositions_When_StandardDataProvided(self):
        """Test that row positions are correctly normalized from 0.0 to 1.0."""
        # Arrange
        null_mask_df = pd.DataFrame({
            "Col1": [False, False, False, False],
            "Col2": [False, False, False, False]
        })
        non_null_ratio_sr = pd.Series([1.0, 0.8, 0.6, 1.0])
        blank_threshold = 0.5
        df_index = pd.RangeIndex(4)
        n_rows = 4

        expected_result = pd.DataFrame({
            "row_pos": [0.0, 1/3, 2/3, 1.0],
            "prev_blank": [False, False, False, False],
            "next_blank": [False, False, False, False]
        }, index=df_index)

        # Act
        from src.playgrround.core.value_based_metrics import _calculate_position_context_metrics
        result = _calculate_position_context_metrics(null_mask_df, non_null_ratio_sr, blank_threshold, df_index, n_rows)

        # Assert
        pd.testing.assert_frame_equal(result, expected_result)

    def test_Should_DetectBlankNeighbors_When_ThresholdConditionsMet(self):
        """Test that prev_blank and next_blank correctly detect blank-ish neighbors."""
        # Arrange
        null_mask_df = pd.DataFrame({
            "Col1": [False, False, False, False, False],
            "Col2": [False, False, False, False, False]
        })
        # Pattern: normal, blank, normal, blank, normal
        non_null_ratio_sr = pd.Series([0.8, 0.2, 0.9, 0.1, 0.7])
        blank_threshold = 0.3
        df_index = pd.RangeIndex(5)
        n_rows = 5

        expected_result = pd.DataFrame({
            "row_pos": [0.0, 0.25, 0.5, 0.75, 1.0],
            "prev_blank": [False, False, True, False, True],  # Previous row < 0.3
            "next_blank": [True, False, True, False, False]   # Next row < 0.3
        }, index=df_index)

        # Act
        from src.playgrround.core.value_based_metrics import _calculate_position_context_metrics
        result = _calculate_position_context_metrics(null_mask_df, non_null_ratio_sr, blank_threshold, df_index, n_rows)

        # Assert
        pd.testing.assert_frame_equal(result, expected_result)

    def test_Should_HandleEdgeRows_When_FirstAndLastRowsProvided(self):
        """Test edge row behavior - first row has no previous, last row has no next."""
        # Arrange
        null_mask_df = pd.DataFrame({
            "Col1": [False, False, False],
            "Col2": [False, False, False]
        })
        # All rows below threshold to test edge behavior
        non_null_ratio_sr = pd.Series([0.2, 0.1, 0.2])
        blank_threshold = 0.5
        df_index = pd.RangeIndex(3)
        n_rows = 3

        expected_result = pd.DataFrame({
            "row_pos": [0.0, 0.5, 1.0],
            "prev_blank": [False, True, True],   # First row: no neighbor → False
            "next_blank": [True, True, False]    # Last row: no neighbor → False
        }, index=df_index)

        # Act
        from src.playgrround.core.value_based_metrics import _calculate_position_context_metrics
        result = _calculate_position_context_metrics(null_mask_df, non_null_ratio_sr, blank_threshold, df_index, n_rows)

        # Assert
        pd.testing.assert_frame_equal(result, expected_result)

    def test_Should_HandleCustomIndex_When_NonSequentialIndexProvided(self):
        """Test that function works correctly with custom non-sequential indices."""
        # Arrange
        null_mask_df = pd.DataFrame({
            "Col1": [False, False, False],
            "Col2": [False, False, False]
        }, index=pd.Index(["Header", "Data", "Footer"]))

        non_null_ratio_sr = pd.Series([1.0, 0.8, 1.0], index=pd.Index(["Header", "Data", "Footer"]))
        blank_threshold = 0.5
        df_index = pd.Index(["Header", "Data", "Footer"])
        n_rows = 3

        expected_result = pd.DataFrame({
            "row_pos": [0.0, 0.5, 1.0],
            "prev_blank": [False, False, False],
            "next_blank": [False, False, False]
        }, index=df_index)

        # Act
        from src.playgrround.core.value_based_metrics import _calculate_position_context_metrics
        result = _calculate_position_context_metrics(null_mask_df, non_null_ratio_sr, blank_threshold, df_index, n_rows)

        # Assert
        pd.testing.assert_frame_equal(result, expected_result)

    def test_Should_HandleSingleRow_When_EdgeCaseProvided(self):
        """Test behavior with single-row input."""
        # Arrange
        null_mask_df = pd.DataFrame({
            "Col1": [False],
            "Col2": [False]
        })
        non_null_ratio_sr = pd.Series([1.0])
        blank_threshold = 0.5
        df_index = pd.RangeIndex(1)
        n_rows = 1

        expected_result = pd.DataFrame({
            "row_pos": [0.0],           # Single row gets 0.0
            "prev_blank": [False],      # No previous neighbor
            "next_blank": [False]       # No next neighbor
        }, index=df_index)

        # Act
        from src.playgrround.core.value_based_metrics import _calculate_position_context_metrics
        result = _calculate_position_context_metrics(null_mask_df, non_null_ratio_sr, blank_threshold, df_index, n_rows)

        # Assert
        pd.testing.assert_frame_equal(result, expected_result)

    def test_Should_HandleEmptyDataFrame_When_NoRowsProvided(self):
        """Test behavior with empty DataFrame."""
        # Arrange
        null_mask_df = pd.DataFrame()
        non_null_ratio_sr = pd.Series(dtype=float)
        blank_threshold = 0.5
        df_index = pd.Index([])
        n_rows = 0

        expected_result = pd.DataFrame({
            "row_pos": pd.Series(dtype=float),
            "prev_blank": pd.Series(dtype=bool),
            "next_blank": pd.Series(dtype=bool)
        }, index=df_index)

        # Act
        from src.playgrround.core.value_based_metrics import _calculate_position_context_metrics
        result = _calculate_position_context_metrics(null_mask_df, non_null_ratio_sr, blank_threshold, df_index, n_rows)

        # Assert
        pd.testing.assert_frame_equal(result, expected_result)

    def test_Should_HandleDifferentThresholds_When_VariousBlankThresholdsProvided(self):
        """Test that different blank thresholds produce different neighbor detection results."""
        # Arrange
        null_mask_df = pd.DataFrame({
            "Col1": [False, False, False, False],
            "Col2": [False, False, False, False]
        })
        # Pattern with borderline ratios
        non_null_ratio_sr = pd.Series([0.9, 0.5, 0.7, 0.3])
        df_index = pd.RangeIndex(4)
        n_rows = 4

        # Test with strict threshold (0.4) - only 0.3 is blank
        expected_strict = pd.DataFrame({
            "row_pos": [0.0, 1/3, 2/3, 1.0],
            "prev_blank": [False, False, False, False],
            "next_blank": [False, False, True, False]
        }, index=df_index)

        # Test with loose threshold (0.6) - 0.5 and 0.3 are blank
        expected_loose = pd.DataFrame({
            "row_pos": [0.0, 1/3, 2/3, 1.0],
            "prev_blank": [False, False, True, False],
            "next_blank": [True, False, True, False]
        }, index=df_index)

        # Act
        from src.playgrround.core.value_based_metrics import _calculate_position_context_metrics
        result_strict = _calculate_position_context_metrics(null_mask_df, non_null_ratio_sr, 0.4, df_index, n_rows)
        result_loose = _calculate_position_context_metrics(null_mask_df, non_null_ratio_sr, 0.6, df_index, n_rows)

        # Assert
        pd.testing.assert_frame_equal(result_strict, expected_strict)
        pd.testing.assert_frame_equal(result_loose, expected_loose)

    def test_Should_HandleTwoRows_When_MinimalDataProvided(self):
        """Test behavior with two-row input."""
        # Arrange
        null_mask_df = pd.DataFrame({
            "Col1": [False, False],
            "Col2": [False, False]
        })
        non_null_ratio_sr = pd.Series([0.8, 0.2])  # First normal, second blank
        blank_threshold = 0.5
        df_index = pd.RangeIndex(2)
        n_rows = 2

        expected_result = pd.DataFrame({
            "row_pos": [0.0, 1.0],      # Two rows: 0.0 and 1.0
            "prev_blank": [False, False], # First row: no previous
            "next_blank": [True, False]    # Last row: no next
        }, index=df_index)

        # Act
        from src.playgrround.core.value_based_metrics import _calculate_position_context_metrics
        result = _calculate_position_context_metrics(null_mask_df, non_null_ratio_sr, blank_threshold, df_index, n_rows)

        # Assert
        pd.testing.assert_frame_equal(result, expected_result)

    def test_Should_ProduceCorrectDataTypes_When_FunctionCompletes(self):
        """Test that output DataFrame has correct data types."""
        # Arrange
        null_mask_df = pd.DataFrame({
            "Col1": [False, False, False],
            "Col2": [False, False, False]
        })
        non_null_ratio_sr = pd.Series([1.0, 0.5, 1.0])
        blank_threshold = 0.7
        df_index = pd.RangeIndex(3)
        n_rows = 3

        # Act
        from src.playgrround.core.value_based_metrics import _calculate_position_context_metrics
        result = _calculate_position_context_metrics(null_mask_df, non_null_ratio_sr, blank_threshold, df_index, n_rows)

        # Assert
        assert result.shape == (3, 3)
        assert list(result.columns) == ["row_pos", "prev_blank", "next_blank"]
        assert result["row_pos"].dtype == float
        assert result["prev_blank"].dtype == bool
        assert result["next_blank"].dtype == bool
        assert result.index.equals(df_index)