import unittest
from decimal import Decimal
from unit_converter import UnitConverter, CurrencyRateProvider, ConversionResult


class _DecimalTestMixin:
    def assertDecimalEqual(self, first, second):
        self.assertEqual(Decimal(str(first)), Decimal(str(second)))


class TestUnitConverter(_DecimalTestMixin, unittest.TestCase):

    def test_length_same_unit(self):
        self.assertDecimalEqual(UnitConverter.convert(100, 'm', 'm'), '100.00')
        self.assertDecimalEqual(UnitConverter.length(5, 'km', 'km'), '5.00')

    def test_length_m_to_km(self):
        self.assertDecimalEqual(UnitConverter.convert(1000, 'm', 'km'), '1.00')
        self.assertDecimalEqual(UnitConverter.length(1500, 'm', 'km'), '1.50')

    def test_length_km_to_m(self):
        self.assertDecimalEqual(UnitConverter.convert(1, 'km', 'm'), '1000.00')
        self.assertDecimalEqual(UnitConverter.length(2.5, 'km', 'm'), '2500.00')

    def test_length_m_to_mi(self):
        self.assertDecimalEqual(UnitConverter.convert(1609.344, 'm', 'mi'), '1.00')
        self.assertDecimalEqual(UnitConverter.length(100, 'm', 'mi', decimal_places=10), '0.0621371192')

    def test_length_mi_to_km(self):
        self.assertDecimalEqual(UnitConverter.convert(1, 'mi', 'km', decimal_places=6), '1.609344')
        self.assertDecimalEqual(UnitConverter.length(10, 'mi', 'km', decimal_places=5), '16.09344')

    def test_weight_g_to_kg(self):
        self.assertDecimalEqual(UnitConverter.convert(1000, 'g', 'kg'), '1.00')
        self.assertDecimalEqual(UnitConverter.weight(1500, 'g', 'kg'), '1.50')

    def test_weight_kg_to_g(self):
        self.assertDecimalEqual(UnitConverter.convert(1, 'kg', 'g'), '1000.00')
        self.assertDecimalEqual(UnitConverter.weight(2.5, 'kg', 'g'), '2500.00')

    def test_weight_lb_to_kg(self):
        self.assertDecimalEqual(UnitConverter.convert(1, 'lb', 'kg', decimal_places=7), '0.4535924')
        self.assertDecimalEqual(UnitConverter.weight(10, 'lb', 'kg', decimal_places=6), '4.535924')

    def test_weight_kg_to_lb(self):
        self.assertDecimalEqual(UnitConverter.convert(1, 'kg', 'lb', decimal_places=6), '2.204623')

    def test_temperature_c_to_f(self):
        self.assertDecimalEqual(UnitConverter.convert(0, 'C', 'F'), '32.00')
        self.assertDecimalEqual(UnitConverter.convert(100, 'c', 'f'), '212.00')
        self.assertDecimalEqual(UnitConverter.temperature(20, 'C', 'F'), '68.00')

    def test_temperature_f_to_c(self):
        self.assertDecimalEqual(UnitConverter.convert(32, 'F', 'C'), '0.00')
        self.assertDecimalEqual(UnitConverter.convert(212, 'f', 'c'), '100.00')
        self.assertDecimalEqual(UnitConverter.temperature(68, 'F', 'C'), '20.00')

    def test_storage_mb_to_gb(self):
        self.assertDecimalEqual(UnitConverter.convert(1024, 'mb', 'gb'), '1.00')
        self.assertDecimalEqual(UnitConverter.storage(2048, 'mb', 'gb'), '2.00')

    def test_storage_gb_to_mb(self):
        self.assertDecimalEqual(UnitConverter.convert(1, 'gb', 'mb'), '1024.00')
        self.assertDecimalEqual(UnitConverter.storage(2, 'gb', 'mb'), '2048.00')

    def test_storage_gb_to_tb(self):
        self.assertDecimalEqual(UnitConverter.convert(1024, 'gb', 'tb'), '1.00')
        self.assertDecimalEqual(UnitConverter.storage(2048, 'gb', 'tb'), '2.00')

    def test_storage_tb_to_mb(self):
        self.assertDecimalEqual(UnitConverter.convert(1, 'tb', 'mb'), '1048576.00')

    def test_unit_case_insensitive(self):
        self.assertDecimalEqual(UnitConverter.convert(1, 'KM', 'M'), '1000.00')
        self.assertDecimalEqual(UnitConverter.convert(1, 'Mi', 'm'), '1609.34')
        self.assertDecimalEqual(UnitConverter.convert(1, 'GB', 'MB'), '1024.00')

    def test_incompatible_units(self):
        with self.assertRaises(ValueError):
            UnitConverter.convert(1, 'm', 'kg')
        with self.assertRaises(ValueError):
            UnitConverter.convert(1, 'c', 'gb')

    def test_unknown_unit(self):
        with self.assertRaises(ValueError):
            UnitConverter.convert(1, 'm', 'xyz')
        with self.assertRaises(ValueError):
            UnitConverter.convert(1, 'abc', 'km')

    def test_get_supported_units(self):
        units = UnitConverter.get_supported_units()
        self.assertIn('length', units)
        self.assertIn('weight', units)
        self.assertIn('temperature', units)
        self.assertIn('storage', units)
        self.assertIn('currency', units)
        self.assertIn('m', units['length'])
        self.assertIn('c', units['temperature'])
        self.assertIn('kg', units['weight'])
        self.assertIn('gb', units['storage'])

    def test_category_methods(self):
        self.assertDecimalEqual(UnitConverter.length(1, 'km', 'm'), '1000.00')
        self.assertDecimalEqual(UnitConverter.weight(1, 'kg', 'g'), '1000.00')
        self.assertDecimalEqual(UnitConverter.temperature(0, 'c', 'f'), '32.00')
        self.assertDecimalEqual(UnitConverter.storage(1, 'gb', 'mb'), '1024.00')

    def test_decimal_places_default(self):
        result = UnitConverter.convert(1, 'mi', 'km')
        self.assertEqual(str(result), '1.61')
        self.assertEqual(len(str(result).split('.')[1]), 2)

    def test_decimal_places_custom(self):
        result = UnitConverter.convert(1, 'mi', 'km', decimal_places=4)
        self.assertEqual(str(result), '1.6093')
        self.assertEqual(len(str(result).split('.')[1]), 4)

    def test_decimal_places_zero(self):
        result = UnitConverter.convert(1.6, 'km', 'm', decimal_places=0)
        self.assertEqual(str(result), '1600')

    def test_decimal_places_negative(self):
        with self.assertRaises(ValueError):
            UnitConverter.convert(1, 'm', 'km', decimal_places=-1)

    def test_float_precision_fix(self):
        float_result = 0.1 + 0.2
        self.assertNotEqual(float_result, 0.3)

        result = UnitConverter.convert(0.1, 'km', 'm', decimal_places=1)
        self.assertDecimalEqual(result, '100.0')

        result2 = UnitConverter.convert(0.2, 'km', 'm', decimal_places=1)
        self.assertDecimalEqual(result2, '200.0')

        result3 = UnitConverter.convert(0.3, 'km', 'm', decimal_places=1)
        self.assertDecimalEqual(result3, '300.0')

    def test_high_precision_calculation(self):
        result = UnitConverter.convert(1, 'lb', 'kg', decimal_places=10)
        self.assertEqual(str(result), '0.4535923700')

        result2 = UnitConverter.convert(1, 'mi', 'm', decimal_places=10)
        self.assertEqual(str(result2), '1609.3440000000')

    def test_string_input(self):
        self.assertDecimalEqual(UnitConverter.convert('1000', 'm', 'km'), '1.00')
        self.assertDecimalEqual(UnitConverter.convert('0.5', 'km', 'm'), '500.00')

    def test_decimal_input(self):
        self.assertDecimalEqual(UnitConverter.convert(Decimal('1.5'), 'km', 'm'), '1500.00')
        self.assertDecimalEqual(UnitConverter.convert(Decimal('0.1'), 'kg', 'g'), '100.00')

    def test_return_type_is_decimal(self):
        result = UnitConverter.convert(1, 'km', 'm')
        self.assertIsInstance(result, Decimal)

    def test_rounding_half_up(self):
        self.assertDecimalEqual(UnitConverter.convert(1.234, 'km', 'm', decimal_places=1), '1234.0')
        self.assertDecimalEqual(UnitConverter.convert(1, 'mi', 'km', decimal_places=1), '1.6')
        self.assertDecimalEqual(UnitConverter.convert(1, 'mi', 'km', decimal_places=2), '1.61')
        self.assertDecimalEqual(UnitConverter.convert(1, 'mi', 'km', decimal_places=3), '1.609')


class TestConversionFormula(_DecimalTestMixin, unittest.TestCase):
    def test_convert_with_formula_returns_result(self):
        result = UnitConverter.convert_with_formula(1, 'km', 'm')
        self.assertIsInstance(result, ConversionResult)
        self.assertIsInstance(result.value, Decimal)
        self.assertIsInstance(result.formula, str)

    def test_formula_linear(self):
        result = UnitConverter.convert_with_formula(1, 'km', 'm')
        self.assertEqual(result.value, Decimal('1000.00'))
        self.assertIn('1 km', result.formula)
        self.assertIn('1000', result.formula)
        self.assertIn('m', result.formula)

    def test_formula_temperature_c_to_f(self):
        result = UnitConverter.convert_with_formula(0, 'c', 'f')
        self.assertEqual(result.value, Decimal('32.00'))
        self.assertIn('9/5', result.formula)
        self.assertIn('32', result.formula)

    def test_formula_temperature_f_to_c(self):
        result = UnitConverter.convert_with_formula(32, 'f', 'c')
        self.assertEqual(result.value, Decimal('0.00'))
        self.assertIn('5/9', result.formula)
        self.assertIn('32', result.formula)

    def test_formula_same_unit(self):
        result = UnitConverter.convert_with_formula(5, 'km', 'km')
        self.assertEqual(result.value, Decimal('5.00'))
        self.assertIn('5', result.formula)
        self.assertIn('km', result.formula)

    def test_formula_currency(self):
        CurrencyRateProvider.set_rates({'USD': 1.0, 'CNY': 7.24})
        result = UnitConverter.convert_with_formula(1, 'USD', 'CNY')
        self.assertDecimalEqual(result.value, '7.24')
        self.assertIn('USD', result.formula)
        self.assertIn('CNY', result.formula)
        self.assertIn('汇率', result.formula)

    def test_formula_preserves_original_case(self):
        result = UnitConverter.convert_with_formula(1, 'KM', 'M')
        self.assertIn('KM', result.formula)
        self.assertIn('M', result.formula)

    def test_formula_weight(self):
        result = UnitConverter.convert_with_formula(1, 'kg', 'g')
        self.assertEqual(result.value, Decimal('1000.00'))
        self.assertIn('1 kg', result.formula)
        self.assertIn('g', result.formula)

    def test_formula_storage(self):
        result = UnitConverter.convert_with_formula(1, 'gb', 'mb')
        self.assertEqual(result.value, Decimal('1024.00'))
        self.assertIn('1 gb', result.formula)
        self.assertIn('mb', result.formula)


class TestCurrencyConversion(_DecimalTestMixin, unittest.TestCase):
    def setUp(self):
        CurrencyRateProvider.set_rates({
            'USD': 1.0,
            'CNY': 7.24,
            'EUR': 0.92,
            'GBP': 0.79,
            'JPY': 154.5,
        })

    def test_usd_to_cny(self):
        result = UnitConverter.currency(1, 'USD', 'CNY')
        self.assertDecimalEqual(result, '7.24')

    def test_cny_to_usd(self):
        result = UnitConverter.currency(7.24, 'CNY', 'USD', decimal_places=2)
        self.assertDecimalEqual(result, '1.00')

    def test_eur_to_cny(self):
        result = UnitConverter.currency(1, 'EUR', 'CNY', decimal_places=2)
        expected = Decimal('7.24') / Decimal('0.92')
        self.assertDecimalEqual(result, str(expected.quantize(Decimal('0.01'))))

    def test_currency_via_convert(self):
        result = UnitConverter.convert(100, 'USD', 'CNY', decimal_places=2)
        self.assertDecimalEqual(result, '724.00')

    def test_currency_case_insensitive(self):
        result = UnitConverter.convert(1, 'usd', 'cny', decimal_places=2)
        self.assertDecimalEqual(result, '7.24')

    def test_currency_same_unit(self):
        result = UnitConverter.convert(100, 'USD', 'USD')
        self.assertDecimalEqual(result, '100.00')

    def test_unsupported_currency(self):
        with self.assertRaises(ValueError):
            UnitConverter.convert(1, 'USD', 'XYZ')

    def test_currency_in_get_supported_units(self):
        units = UnitConverter.get_supported_units()
        self.assertIn('currency', units)
        self.assertIn('USD', units['currency'])
        self.assertIn('CNY', units['currency'])


class TestCurrencyRateProvider(unittest.TestCase):
    def setUp(self):
        CurrencyRateProvider.set_rates({
            'USD': 1.0,
            'CNY': 7.24,
            'EUR': 0.92,
        })

    def test_get_rate(self):
        rate = CurrencyRateProvider.get_rate('CNY')
        self.assertEqual(rate, Decimal('7.24'))

    def test_get_rate_usd_base(self):
        rate = CurrencyRateProvider.get_rate('USD')
        self.assertEqual(rate, Decimal('1.0'))

    def test_get_rate_unknown(self):
        rate = CurrencyRateProvider.get_rate('XYZ')
        self.assertIsNone(rate)

    def test_get_rates(self):
        rates = CurrencyRateProvider.get_rates()
        self.assertIn('USD', rates)
        self.assertIn('CNY', rates)
        self.assertEqual(rates['CNY'], Decimal('7.24'))

    def test_set_rates(self):
        CurrencyRateProvider.set_rates({'USD': 1.0, 'AUD': 1.53})
        rate = CurrencyRateProvider.get_rate('AUD')
        self.assertEqual(rate, Decimal('1.53'))

    def test_is_cache_expired_after_set(self):
        CurrencyRateProvider.set_rates({'USD': 1.0, 'CNY': 7.24})
        self.assertFalse(CurrencyRateProvider.is_cache_expired())

    def test_get_supported_currencies(self):
        currencies = CurrencyRateProvider.get_supported_currencies()
        self.assertIn('USD', currencies)
        self.assertEqual(currencies['CNY'], '人民币')

    def test_default_rates_available(self):
        CurrencyRateProvider._rates = {}
        CurrencyRateProvider._last_fetch = 0
        rates = CurrencyRateProvider._get_default_rates()
        self.assertIn('USD', rates)
        self.assertIn('CNY', rates)
        self.assertIn('EUR', rates)
        self.assertIn('JPY', rates)

    def test_set_rates_always_includes_usd(self):
        CurrencyRateProvider.set_rates({'CNY': 7.24})
        self.assertEqual(CurrencyRateProvider.get_rate('USD'), Decimal('1.0'))

    def tearDown(self):
        CurrencyRateProvider.set_rates({
            'USD': 1.0,
            'CNY': 7.24,
            'EUR': 0.92,
            'GBP': 0.79,
            'JPY': 154.5,
        })


class TestCustomUnitRegistration(_DecimalTestMixin, unittest.TestCase):
    def tearDown(self):
        if 'area' in UnitConverter._CUSTOM_CATEGORIES:
            del UnitConverter._CUSTOM_CATEGORIES['area']
        if 'area' in UnitConverter._CUSTOM_UNIT_NAMES:
            del UnitConverter._CUSTOM_UNIT_NAMES['area']
        if 'area' in UnitConverter._CATEGORIES:
            del UnitConverter._CATEGORIES['area']
        if 'ft' in UnitConverter._LENGTH:
            del UnitConverter._LENGTH['ft']
        UnitConverter._CATEGORIES['length'] = tuple(
            u for u in UnitConverter._CATEGORIES['length'] if u != 'ft'
        )
        if 'ft' in UnitConverter._UNIT_NAMES.get('length', {}):
            del UnitConverter._UNIT_NAMES['length']['ft']

    def test_register_new_category(self):
        UnitConverter.register_unit('area', 'sqm', 1.0, '平方米')
        UnitConverter.register_unit('area', 'sqkm', 1000000.0, '平方千米')

        result = UnitConverter.convert(1, 'sqkm', 'sqm', decimal_places=0)
        self.assertDecimalEqual(result, '1000000')

    def test_register_new_category_formula(self):
        UnitConverter.register_unit('area', 'sqm', 1.0, '平方米')
        UnitConverter.register_unit('area', 'sqkm', 1000000.0, '平方千米')

        result = UnitConverter.convert_with_formula(1, 'sqkm', 'sqm')
        self.assertEqual(result.value, Decimal('1000000.00'))
        self.assertIn('1 sqkm', result.formula)

    def test_register_to_existing_category(self):
        UnitConverter.register_unit('length', 'ft', 0.3048, '英尺')

        result = UnitConverter.convert(1, 'ft', 'm', decimal_places=4)
        self.assertDecimalEqual(result, '0.3048')

    def test_register_to_existing_category_formula(self):
        UnitConverter.register_unit('length', 'ft', 0.3048, '英尺')

        result = UnitConverter.convert_with_formula(1, 'ft', 'm', decimal_places=4)
        self.assertIn('1 ft', result.formula)
        self.assertIn('m', result.formula)

    def test_unregister_from_existing_category(self):
        UnitConverter.register_unit('length', 'ft', 0.3048, '英尺')
        self.assertIn('ft', UnitConverter._LENGTH)

        UnitConverter.unregister_unit('length', 'ft')
        self.assertNotIn('ft', UnitConverter._LENGTH)

    def test_unregister_custom_category(self):
        UnitConverter.register_unit('area', 'sqm', 1.0, '平方米')
        UnitConverter.register_unit('area', 'sqkm', 1000000.0, '平方千米')

        UnitConverter.unregister_unit('area', 'sqm')

        with self.assertRaises(ValueError):
            UnitConverter.convert(1, 'sqm', 'sqkm')

    def test_unregister_last_unit_removes_category(self):
        UnitConverter.register_unit('area', 'sqm', 1.0, '平方米')

        UnitConverter.unregister_unit('area', 'sqm')
        self.assertNotIn('area', UnitConverter._CUSTOM_CATEGORIES)

    def test_register_temperature_raises(self):
        with self.assertRaises(ValueError):
            UnitConverter.register_unit('temperature', 'k', 1.0, '开尔文')

    def test_register_currency_raises(self):
        with self.assertRaises(ValueError):
            UnitConverter.register_unit('currency', 'BTC', 50000.0, '比特币')

    def test_unregister_temperature_raises(self):
        with self.assertRaises(ValueError):
            UnitConverter.unregister_unit('temperature', 'c')

    def test_unregister_currency_raises(self):
        with self.assertRaises(ValueError):
            UnitConverter.unregister_unit('currency', 'USD')

    def test_register_display_name_in_supported_units(self):
        UnitConverter.register_unit('area', 'sqm', 1.0, '平方米')
        UnitConverter.register_unit('area', 'sqkm', 1000000.0, '平方千米')

        units = UnitConverter.get_supported_units()
        self.assertIn('area', units)
        self.assertEqual(units['area']['sqm'], '平方米')
        self.assertEqual(units['area']['sqkm'], '平方千米')

    def test_register_without_display_name(self):
        UnitConverter.register_unit('area', 'sqm', 1.0)

        units = UnitConverter.get_supported_units()
        self.assertIn('area', units)
        self.assertIn('sqm', units['area'])

    def test_unregister_unknown_category_raises(self):
        with self.assertRaises(ValueError):
            UnitConverter.unregister_unit('nonexistent', 'xyz')

    def test_custom_unit_convert_back_and_forth(self):
        UnitConverter.register_unit('area', 'sqm', 1.0, '平方米')
        UnitConverter.register_unit('area', 'sqkm', 1000000.0, '平方千米')

        result1 = UnitConverter.convert(1, 'sqkm', 'sqm', decimal_places=0)
        self.assertDecimalEqual(result1, '1000000')

        result2 = UnitConverter.convert(1000000, 'sqm', 'sqkm', decimal_places=2)
        self.assertDecimalEqual(result2, '1.00')


if __name__ == '__main__':
    unittest.main()
