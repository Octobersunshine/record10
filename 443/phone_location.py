import csv
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timedelta
from functools import lru_cache

try:
    import urllib.request
    import urllib.error
    HAS_URLLIB = True
except ImportError:
    HAS_URLLIB = False


BATCH_MAX_SIZE = 100


class MnpChecker:
    PORTABILITY_APIS = [
        {
            "name": "天行携号转网API",
            "url": "https://apis.tianapi.com/mobile/index?key={api_key}&mobile={phone}",
            "format": "json",
            "carrier_field": "result.operator",
            "province_field": "result.province",
            "city_field": "result.city"
        }
    ]

    def __init__(self, data_dir=None):
        if data_dir is None:
            data_dir = os.path.dirname(os.path.abspath(__file__))
        self.mnp_cache_path = os.path.join(data_dir, "mnp_cache.json")
        self._mnp_cache = {}
        self._load_mnp_cache()

    def _load_mnp_cache(self):
        if os.path.isfile(self.mnp_cache_path):
            try:
                with open(self.mnp_cache_path, "r", encoding="utf-8") as f:
                    self._mnp_cache = json.load(f)
            except Exception:
                self._mnp_cache = {}

    def _save_mnp_cache(self):
        try:
            with open(self.mnp_cache_path, "w", encoding="utf-8") as f:
                json.dump(self._mnp_cache, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    @staticmethod
    def _normalize_carrier(raw_carrier):
        if not raw_carrier:
            return "未知"
        if "移动" in raw_carrier:
            return "中国移动"
        if "联通" in raw_carrier:
            return "中国联通"
        if "电信" in raw_carrier:
            return "中国电信"
        return raw_carrier

    def check_portability(self, phone, original_carrier, api_key=None):
        if not HAS_URLLIB:
            return {"ported": False, "actual_carrier": original_carrier, "source": "offline"}

        if not api_key:
            api_key = os.environ.get("TIANAPI_KEY", "")
        if not api_key:
            return {"ported": False, "actual_carrier": original_carrier, "source": "no_api_key"}

        phone_clean = phone.strip().replace("-", "").replace(" ", "")
        if phone_clean in self._mnp_cache:
            cached = self._mnp_cache[phone_clean]
            if "expire_at" in cached:
                if datetime.fromisoformat(cached["expire_at"]) > datetime.now():
                    actual = cached.get("actual_carrier", original_carrier)
                    return {
                        "ported": actual != original_carrier,
                        "actual_carrier": actual,
                        "original_carrier": original_carrier,
                        "source": cached.get("source", "cache")
                    }
                else:
                    del self._mnp_cache[phone_clean]

        api = self.PORTABILITY_APIS[0]
        url = api["url"].format(api_key=api_key, phone=phone_clean)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if data.get("code") == 200:
                    result = data.get("result", {})
                    actual_carrier = self._normalize_carrier(result.get("operator", ""))
                    ported = (actual_carrier != original_carrier and actual_carrier != "未知")

                    self._mnp_cache[phone_clean] = {
                        "actual_carrier": actual_carrier,
                        "original_carrier": original_carrier,
                        "ported": ported,
                        "source": "online_api",
                        "expire_at": (datetime.now() + timedelta(days=7)).isoformat()
                    }
                    self._save_mnp_cache()

                    return {
                        "ported": ported,
                        "actual_carrier": actual_carrier,
                        "original_carrier": original_carrier,
                        "source": "online_api"
                    }
        except Exception:
            pass

        return {"ported": False, "actual_carrier": original_carrier, "source": "query_failed"}


class QueryStats:
    def __init__(self, results):
        self.results = results
        self.valid_results = [r for r in results if "error" not in r]
        self.error_results = [r for r in results if "error" in r]

    def carrier_distribution(self):
        counter = Counter(r.get("actual_carrier") or r.get("carrier", "未知") for r in self.valid_results)
        total = sum(counter.values())
        distribution = {}
        for carrier, count in counter.most_common():
            distribution[carrier] = {
                "count": count,
                "percentage": round(count / total * 100, 1) if total > 0 else 0
            }
        return distribution

    def province_distribution(self, top_n=10):
        counter = Counter(r.get("province", "未知") for r in self.valid_results)
        total = sum(counter.values())
        distribution = {}
        for province, count in counter.most_common(top_n):
            distribution[province] = {
                "count": count,
                "percentage": round(count / total * 100, 1) if total > 0 else 0
            }
        return distribution

    def portability_stats(self):
        ported = sum(1 for r in self.valid_results if r.get("ported", False))
        not_ported = sum(1 for r in self.valid_results if not r.get("ported", False) and r.get("mnp_source") != "no_api_key")
        unchecked = sum(1 for r in self.valid_results if r.get("mnp_source") in ("no_api_key", None))
        total = len(self.valid_results)
        return {
            "ported": ported,
            "ported_percentage": round(ported / total * 100, 1) if total > 0 else 0,
            "not_ported": not_ported,
            "unchecked": unchecked,
            "total": total
        }

    def data_source_stats(self):
        counter = Counter(r.get("data_source", "未知") for r in self.valid_results)
        total = sum(counter.values())
        return {
            source: {"count": count, "percentage": round(count / total * 100, 1) if total > 0 else 0}
            for source, count in counter.most_common()
        }

    def full_report(self):
        return {
            "total_input": len(self.results),
            "valid_count": len(self.valid_results),
            "error_count": len(self.error_results),
            "carrier_distribution": self.carrier_distribution(),
            "province_distribution": self.province_distribution(),
            "portability_stats": self.portability_stats(),
            "data_source_stats": self.data_source_stats()
        }

    def print_report(self):
        report = self.full_report()
        print("=" * 60)
        print("              📊 批量查询统计报告")
        print("=" * 60)

        print(f"\n📋 查询总览")
        print(f"   输入号码数: {report['total_input']}")
        print(f"   有效结果数: {report['valid_count']}")
        print(f"   错误号码数: {report['error_count']}")

        print(f"\n🏢 运营商分布 (饼图数据)")
        for carrier, info in report["carrier_distribution"].items():
            bar = "█" * int(info["percentage"] / 2)
            print(f"   {carrier:8s} {info['count']:4d}个 ({info['percentage']:5.1f}%) {bar}")

        print(f"\n📍 省份分布 TOP-10")
        for province, info in report["province_distribution"].items():
            bar = "▓" * int(info["percentage"] / 2)
            print(f"   {province:8s} {info['count']:4d}个 ({info['percentage']:5.1f}%) {bar}")

        mnp = report["portability_stats"]
        print(f"\n🔄 携号转网统计")
        print(f"   已转网:   {mnp['ported']}个 ({mnp['ported_percentage']}%)")
        print(f"   未转网:   {mnp['not_ported']}个")
        print(f"   未检测:   {mnp['unchecked']}个")

        print(f"\n📡 数据来源统计")
        for source, info in report["data_source_stats"].items():
            source_label = {"local_db": "本地数据库", "online_api": "在线API",
                           "rule_infer": "规则推断", "cache": "缓存"}.get(source, source)
            print(f"   {source_label:10s} {info['count']:4d}个 ({info['percentage']:5.1f}%)")

        print("\n" + "=" * 60)


class PhoneSegmentDB:
    CARRIER_RULES = {
        "1": {"30": "中国联通", "31": "中国联通", "32": "中国联通",
              "45": "中国联通", "55": "中国联通", "56": "中国联通",
              "85": "中国联通", "86": "中国联通",
              "66": "中国联通", "67": "中国联通",
              "33": "中国电信", "49": "中国电信", "53": "中国电信",
              "73": "中国电信", "77": "中国电信", "80": "中国电信",
              "81": "中国电信", "89": "中国电信", "99": "中国电信",
              "91": "中国电信", "93": "中国电信",
              "34": "中国移动", "35": "中国移动", "36": "中国移动",
              "37": "中国移动", "38": "中国移动", "39": "中国移动",
              "47": "中国移动", "50": "中国移动", "51": "中国移动",
              "52": "中国移动", "57": "中国移动", "58": "中国移动",
              "59": "中国移动", "78": "中国移动", "82": "中国移动",
              "83": "中国移动", "84": "中国移动", "87": "中国移动",
              "88": "中国移动", "98": "中国移动",
              "72": "中国移动", "70": "中国移动"},
    }

    UPDATE_APIS = [
        {
            "name": "手机号段API",
            "url": "https://apis.tianapi.com/mobile/index?key={api_key}&mobile={prefix}",
            "format": "json",
            "response_path": ["result"],
            "fields_map": {
                "prefix": "prefix",
                "carrier": "operator",
                "province": "province",
                "city": "city"
            }
        }
    ]

    def __init__(self, data_dir=None):
        if data_dir is None:
            data_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = data_dir
        self.csv_path = os.path.join(data_dir, "phone_segments.csv")
        self.meta_path = os.path.join(data_dir, "phone_segments_meta.json")
        self.cache_path = os.path.join(data_dir, "phone_segments_cache.json")

        self._data = {}
        self._cache = {}
        self._meta = {
            "version": "1.0.0",
            "last_updated": "2024-01-01T00:00:00",
            "total_segments": 0,
            "data_source": "local"
        }

        self.mnp_checker = MnpChecker(data_dir)

        self._load_meta()
        self._load_cache()
        self._load_csv()

    def _load_meta(self):
        if os.path.isfile(self.meta_path):
            try:
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    self._meta.update(json.load(f))
            except Exception:
                pass

    def _save_meta(self):
        try:
            with open(self.meta_path, "w", encoding="utf-8") as f:
                json.dump(self._meta, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _load_cache(self):
        if os.path.isfile(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
            except Exception:
                self._cache = {}

    def _save_cache(self):
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _load_csv(self):
        if not os.path.isfile(self.csv_path):
            print(f"[警告] 号段数据文件不存在: {self.csv_path}")
            return
        count = 0
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                prefix = row["prefix"].strip()
                self._data[prefix] = {
                    "carrier": row["carrier"].strip(),
                    "province": row["province"].strip(),
                    "city": row["city"].strip(),
                    "source": "local_db"
                }
                count += 1
        self._meta["total_segments"] = count

    def _save_csv(self):
        fieldnames = ["prefix", "carrier", "province", "city"]
        rows = [{"prefix": k, "carrier": v["carrier"], "province": v["province"], "city": v["city"]}
                for k, v in self._data.items()]
        rows.sort(key=lambda x: x["prefix"])
        with open(self.csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        self._meta["total_segments"] = len(rows)
        self._meta["last_updated"] = datetime.now().isoformat()
        self._save_meta()

    @staticmethod
    def _infer_carrier(phone_prefix):
        if len(phone_prefix) < 3:
            return "未知"
        key = phone_prefix[1:3]
        return PhoneSegmentDB.CARRIER_RULES.get("1", {}).get(key, "未知")

    @lru_cache(maxsize=1000)
    def _lookup_prefix(self, prefix):
        result = self._data.get(prefix)
        if result:
            return dict(result)

        if prefix in self._cache:
            cached = self._cache[prefix]
            if "expire_at" in cached:
                if datetime.fromisoformat(cached["expire_at"]) > datetime.now():
                    return {k: v for k, v in cached.items() if k != "expire_at"}
                else:
                    del self._cache[prefix]
        return None

    def get_version_info(self):
        last_updated = datetime.fromisoformat(self._meta["last_updated"])
        days_old = (datetime.now() - last_updated).days
        is_outdated = days_old > 30
        return {
            "version": self._meta["version"],
            "last_updated": self._meta["last_updated"],
            "days_old": days_old,
            "is_outdated": is_outdated,
            "total_segments": self._meta["total_segments"],
            "cached_segments": len(self._cache)
        }

    def query_online(self, prefix, api_key=None):
        if not HAS_URLLIB:
            return None

        if not api_key:
            api_key = os.environ.get("TIANAPI_KEY", "")

        if not api_key:
            return None

        url = self.UPDATE_APIS[0]["url"].format(api_key=api_key, prefix=prefix)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if data.get("code") == 200:
                    result = data.get("result", {})
                    if result:
                        carrier = result.get("operator", "")
                        if carrier and "移动" in carrier:
                            carrier = "中国移动"
                        elif carrier and "联通" in carrier:
                            carrier = "中国联通"
                        elif carrier and "电信" in carrier:
                            carrier = "中国电信"

                        online_data = {
                            "carrier": carrier or self._infer_carrier(prefix),
                            "province": result.get("province", "未知"),
                            "city": result.get("city", "未知"),
                            "source": "online_api"
                        }

                        if online_data["province"] != "未知":
                            self._data[prefix] = {
                                "carrier": online_data["carrier"],
                                "province": online_data["province"],
                                "city": online_data["city"],
                                "source": "local_db"
                            }
                            self._save_csv()

                        self._cache[prefix] = {
                            **online_data,
                            "expire_at": (datetime.now() + timedelta(days=7)).isoformat()
                        }
                        self._save_cache()
                        return online_data
        except Exception:
            pass
        return None

    def query(self, phone, try_online=True, api_key=None, check_mnp=False):
        phone = phone.strip().replace("-", "").replace(" ", "")
        if not phone.isdigit():
            return {"carrier": "未知", "province": "未知", "city": "未知",
                    "error": "手机号格式不正确，只能包含数字"}
        if len(phone) < 7:
            return {"carrier": "未知", "province": "未知", "city": "未知",
                    "error": "手机号长度不足7位，无法识别归属地"}

        prefix = phone[:7]
        version_info = self.get_version_info()

        result = self._lookup_prefix(prefix)
        if result:
            base = {
                "prefix": prefix,
                "carrier": result["carrier"],
                "province": result["province"],
                "city": result["city"],
                "data_source": result.get("source", "local_db"),
                "is_latest": not version_info["is_outdated"],
                "db_version": version_info["version"],
                "db_updated_days_ago": version_info["days_old"]
            }
        elif try_online:
            online_result = self.query_online(prefix, api_key)
            if online_result:
                base = {
                    "prefix": prefix,
                    "carrier": online_result["carrier"],
                    "province": online_result["province"],
                    "city": online_result["city"],
                    "data_source": online_result["source"],
                    "is_latest": True,
                    "db_version": version_info["version"],
                    "db_updated_days_ago": version_info["days_old"],
                    "note": "数据来自在线API，已自动入库"
                }
            else:
                carrier = self._infer_carrier(prefix)
                base = {
                    "prefix": prefix,
                    "carrier": carrier,
                    "province": "未知",
                    "city": "未知",
                    "data_source": "rule_infer",
                    "is_latest": not version_info["is_outdated"],
                    "db_version": version_info["version"],
                    "db_updated_days_ago": version_info["days_old"],
                    "note": "该号段未在本地数据库中，运营商为规则推断。设置API Key可在线查询详细归属地"
                }
        else:
            carrier = self._infer_carrier(prefix)
            base = {
                "prefix": prefix,
                "carrier": carrier,
                "province": "未知",
                "city": "未知",
                "data_source": "rule_infer",
                "is_latest": not version_info["is_outdated"],
                "db_version": version_info["version"],
                "db_updated_days_ago": version_info["days_old"],
                "note": "该号段未在本地数据库中，运营商为规则推断。设置API Key可在线查询详细归属地"
            }

        if check_mnp:
            mnp_result = self.mnp_checker.check_portability(phone, base["carrier"], api_key)
            base["original_carrier"] = mnp_result.get("original_carrier", base["carrier"])
            base["actual_carrier"] = mnp_result.get("actual_carrier", base["carrier"])
            base["ported"] = mnp_result.get("ported", False)
            base["mnp_source"] = mnp_result.get("source", "unknown")
            if base["ported"]:
                base["note"] = (base.get("note", "") + " | 携号转网: {} → {}".format(
                    base["original_carrier"], base["actual_carrier"])).strip(" |")

        return base

    def batch_query(self, phones, try_online=True, api_key=None,
                    check_mnp=False, show_progress=False, max_size=BATCH_MAX_SIZE):
        if len(phones) > max_size:
            raise ValueError(f"批量查询最多支持 {max_size} 个号码，当前输入 {len(phones)} 个")

        results = []
        total = len(phones)
        for i, phone in enumerate(phones):
            if show_progress and total > 1:
                pct = (i + 1) / total * 100
                sys.stderr.write(f"\r  查询进度: {i + 1}/{total} ({pct:.0f}%)")
                sys.stderr.flush()
            results.append(self.query(phone, try_online=try_online,
                                     api_key=api_key, check_mnp=check_mnp))
            if try_online or check_mnp:
                time.sleep(0.1)
        if show_progress and total > 1:
            sys.stderr.write("\n")
        return results

    def batch_query_with_stats(self, phones, try_online=True, api_key=None,
                               check_mnp=False, show_progress=False,
                               max_size=BATCH_MAX_SIZE):
        results = self.batch_query(phones, try_online=try_online, api_key=api_key,
                                   check_mnp=check_mnp, show_progress=show_progress,
                                   max_size=max_size)
        stats = QueryStats(results)
        return results, stats

    def update_from_online(self, prefix_list, api_key=None):
        if not api_key:
            api_key = os.environ.get("TIANAPI_KEY", "")
        if not api_key:
            return {"success": False, "message": "需要设置 TIANAPI_KEY 环境变量"}

        updated = 0
        failed = 0
        for prefix in prefix_list:
            result = self.query_online(prefix, api_key)
            if result and result.get("province") != "未知":
                updated += 1
            else:
                failed += 1
            time.sleep(0.2)

        return {"success": True, "updated": updated, "failed": failed}

    def interactive(self):
        version_info = self.get_version_info()
        print("=" * 55)
        print("           手机号归属地查询系统 v" + version_info["version"])
        print("=" * 55)
        print(f"  数据库版本: {version_info['version']}")
        print(f"  最后更新: {version_info['last_updated'][:10]} ({version_info['days_old']}天前)")
        print(f"  本地号段数: {version_info['total_segments']}")
        print(f"  缓存号段数: {version_info['cached_segments']}")
        if version_info["is_outdated"]:
            print("  ⚠️  数据库已超过30天未更新，建议更新")
        print("-" * 55)
        print("  输入手机号查询归属地")
        print("  输入 batch 进入批量查询模式")
        print("  输入 update 更新号段库")
        print("  输入 q 退出\n")

        while True:
            try:
                phone = input("请输入手机号: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见！")
                break
            if phone.lower() == "q":
                print("再见！")
                break
            if phone.lower() == "update":
                print("  如需在线更新，请设置环境变量 TIANAPI_KEY\n")
                continue
            if phone.lower() == "batch":
                self._interactive_batch()
                continue
            result = self.query(phone)
            if "error" in result:
                print(f"  ❌ {result['error']}")
            else:
                self._print_single_result(result)
            print()

    def _interactive_batch(self):
        print("\n  📦 批量查询模式 (最多100个号码)")
        print("  输入手机号，用空格/逗号/换行分隔")
        print("  输入 ok 开始查询，输入 cancel 取消\n")
        phones = []
        while True:
            try:
                line = input("  > ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if line.lower() == "cancel":
                print("  已取消批量查询\n")
                return
            if line.lower() == "ok":
                break
            parts = line.replace(",", " ").replace("，", " ").split()
            for p in parts:
                p = p.strip()
                if p:
                    phones.append(p)

        if not phones:
            print("  未输入任何号码\n")
            return
        if len(phones) > BATCH_MAX_SIZE:
            print(f"  ❌ 超过最大限制 {BATCH_MAX_SIZE} 个，当前 {len(phones)} 个\n")
            return

        api_key = os.environ.get("TIANAPI_KEY", "")
        has_key = bool(api_key)
        check_mnp = has_key
        print(f"\n  正在查询 {len(phones)} 个号码...")
        if has_key:
            print("  已检测到 TIANAPI_KEY，将自动查询携号转网信息")
        else:
            print("  未设置 TIANAPI_KEY，仅使用本地数据查询")

        try:
            results, stats = self.batch_query_with_stats(
                phones, try_online=has_key, api_key=api_key,
                check_mnp=check_mnp, show_progress=True)
        except ValueError as e:
            print(f"  ❌ {e}\n")
            return

        print()
        for phone, result in zip(phones, results):
            print(f"  {phone}: ", end="")
            if "error" in result:
                print(f"❌ {result['error']}")
            else:
                carrier = result.get("actual_carrier") or result.get("carrier", "未知")
                ported_mark = " [转网]" if result.get("ported") else ""
                print(f"{carrier}{ported_mark} | {result['province']} {result['city']}")
        print()
        stats.print_report()
        print()

    @staticmethod
    def _print_single_result(result):
        print(f"  📱 号段:   {result['prefix']}")
        carrier = result.get("actual_carrier") or result.get("carrier", "未知")
        original = result.get("original_carrier")
        if result.get("ported") and original:
            print(f"  🏢 运营商: {carrier} (原: {original}，已携号转网)")
        else:
            print(f"  🏢 运营商: {carrier}")
        print(f"  📍 省份:   {result['province']}")
        print(f"  🏙️  城市:   {result['city']}")
        print(f"  📡 数据源: {result['data_source']}")
        if result.get("is_latest"):
            print(f"  ✅ 数据状态: 最新")
        else:
            print(f"  ⚠️  数据状态: 数据库较旧")
        if "note" in result:
            print(f"  ℹ️  {result['note']}")


def main():
    db = PhoneSegmentDB()

    if len(sys.argv) > 1:
        if sys.argv[1] == "--version":
            v = db.get_version_info()
            print(f"版本: {v['version']}")
            print(f"最后更新: {v['last_updated']}")
            print(f"本地号段数: {v['total_segments']}")
            sys.exit(0)
        elif sys.argv[1] == "--update":
            print("在线更新功能需要 API Key")
            print("请设置环境变量: set TIANAPI_KEY=your_api_key")
            sys.exit(0)

        phone_args = []
        check_mnp = False
        show_stats = False
        for arg in sys.argv[1:]:
            if arg == "--mnp":
                check_mnp = True
            elif arg == "--stats":
                show_stats = True
            elif not arg.startswith("--"):
                phone_args.append(arg)

        if phone_args:
            api_key = os.environ.get("TIANAPI_KEY", "")
            has_key = bool(api_key)
            if check_mnp and not has_key:
                check_mnp = False
                print("[提示] 携号转网查询需要设置 TIANAPI_KEY 环境变量\n")

            try:
                if show_stats:
                    results, stats = db.batch_query_with_stats(
                        phone_args, try_online=has_key, api_key=api_key,
                        check_mnp=check_mnp, show_progress=True)
                else:
                    results = db.batch_query(
                        phone_args, try_online=has_key, api_key=api_key,
                        check_mnp=check_mnp, show_progress=True)
            except ValueError as e:
                print(f"❌ {e}")
                sys.exit(1)

            for phone, result in zip(phone_args, results):
                print(f"手机号: {phone}")
                if "error" in result:
                    print(f"  错误: {result['error']}")
                else:
                    print(f"  号段: {result['prefix']}")
                    carrier = result.get("actual_carrier") or result.get("carrier", "未知")
                    original = result.get("original_carrier")
                    if result.get("ported") and original:
                        print(f"  运营商: {carrier} (原: {original}，已携号转网)")
                    else:
                        print(f"  运营商: {carrier}")
                    print(f"  省份: {result['province']}")
                    print(f"  城市: {result['city']}")
                    print(f"  数据源: {result['data_source']}")
                    print(f"  是否最新: {'是' if result.get('is_latest') else '否'}")
                    if "note" in result:
                        print(f"  备注: {result['note']}")
                print()

            if show_stats:
                stats.print_report()
    else:
        db.interactive()


if __name__ == "__main__":
    main()
