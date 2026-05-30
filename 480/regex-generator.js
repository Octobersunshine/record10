const { Worker, isMainThread, parentPort, workerData } = require("worker_threads");

if (!isMainThread) {
  try {
    const result = generateRegexFromSamples(workerData);
    parentPort.postMessage({ success: true, ...result });
  } catch (e) {
    parentPort.postMessage({ success: false, error: e.message });
  }
  return;
}

function classifyChar(ch) {
  if (/[0-9]/.test(ch)) return "\\d";
  if (/[a-z]/.test(ch)) return "[a-z]";
  if (/[A-Z]/.test(ch)) return "[A-Z]";
  if (/[a-zA-Z]/.test(ch)) return "[a-zA-Z]";
  if (/\s/.test(ch)) return "\\s";
  if (/[0-9a-zA-Z_]/.test(ch)) return "\\w";
  if (["(", ")", "[", "]", "{", "}", ".", "*", "+", "?", "|", "^", "$", "\\", "/"].includes(ch)) {
    return "\\" + ch;
  }
  return ch;
}

function escapeRegex(str) {
  return str.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function findLongestCommonPrefix(strs) {
  if (strs.length === 0) return "";
  let prefix = strs[0];
  for (let i = 1; i < strs.length; i++) {
    while (strs[i].indexOf(prefix) !== 0) {
      prefix = prefix.slice(0, -1);
      if (prefix === "") return "";
    }
  }
  return prefix;
}

function findLongestCommonSuffix(strs) {
  if (strs.length === 0) return "";
  const reversed = strs.map(s => s.split("").reverse().join(""));
  const prefix = findLongestCommonPrefix(reversed);
  return prefix.split("").reverse().join("");
}

function tokenize(str) {
  const tokens = [];
  let i = 0;
  while (i < str.length) {
    const ch = str[i];
    const type = classifyChar(ch);
    let j = i + 1;
    while (j < str.length && classifyChar(str[j]) === type) {
      j++;
    }
    tokens.push({ type, value: str.slice(i, j), length: j - i });
    i = j;
  }
  return tokens;
}

function alignTokens(tokenLists) {
  if (tokenLists.length === 0) return [];
  const maxLen = Math.max(...tokenLists.map(t => t.length));
  const aligned = [];

  for (let pos = 0; pos < maxLen; pos++) {
    const typesAtPos = new Set();
    const allSame = tokenLists.every(tl => tl[pos] && tokenLists[0][pos] && tl[pos].type === tokenLists[0][pos].type);

    if (allSame) {
      const type = tokenLists[0][pos].type;
      const lengths = tokenLists.map(tl => tl[pos]?.length || 0).filter(l => l > 0);
      const minLen = Math.min(...lengths);
      const maxLen2 = Math.max(...lengths);

      let quantifier;
      if (minLen === maxLen2) {
        quantifier = minLen === 1 ? "" : `{${minLen}}`;
      } else {
        quantifier = minLen === 1 && maxLen2 > 1 ? "+" : "*";
      }
      aligned.push({ type, quantifier, isFixed: type.length === 1 && !type.startsWith("\\") && !type.startsWith("[") });
    } else {
      const allTypes = new Set();
      tokenLists.forEach(tl => {
        if (tl[pos]) allTypes.add(tl[pos].type);
      });
      if (allTypes.size > 0) {
        aligned.push({ type: ".", quantifier: "+", isFixed: false });
      }
    }
  }
  return aligned;
}

function generatePatternFromTokens(tokens) {
  let pattern = "";
  for (const t of tokens) {
    if (t.isFixed) {
      pattern += t.type.replace(/\\(.)/g, "$1");
    } else {
      pattern += t.type + t.quantifier;
    }
  }
  return pattern;
}

function generateCandidates(positives) {
  const candidates = [];

  if (positives.length === 0) return candidates;

  const prefix = findLongestCommonPrefix(positives);
  const suffix = findLongestCommonSuffix(positives);

  if (prefix && prefix.length > 0) {
    candidates.push({ pattern: "^" + escapeRegex(prefix), type: "prefix_match", score: 60 });
  }

  if (suffix && suffix.length > 0) {
    candidates.push({ pattern: escapeRegex(suffix) + "$", type: "suffix_match", score: 60 });
  }

  if (prefix && suffix && prefix.length > 0 && suffix.length > 0) {
    const middle = "[\\s\\S]*";
    candidates.push({
      pattern: "^" + escapeRegex(prefix) + middle + escapeRegex(suffix) + "$",
      type: "prefix_middle_suffix",
      score: 80,
    });
  }

  const tokenLists = positives.map(tokenize);
  const aligned = alignTokens(tokenLists);
  if (aligned.length > 0) {
    const pattern = generatePatternFromTokens(aligned);
    if (pattern && pattern.length > 0) {
      candidates.push({ pattern: "^" + pattern + "$", type: "token_align", score: 85 });
    }
  }

  if (positives.length === 1) {
    const exact = positives[0];
    candidates.push({ pattern: "^" + escapeRegex(exact) + "$", type: "exact_match", score: 95 });

    const loose = exact.split("").map(ch => {
      const t = classifyChar(ch);
      return t.length === 1 && !t.startsWith("\\") && !t.startsWith("[") ? escapeRegex(ch) : t + "+";
    }).join("");
    if (loose !== exact) {
      candidates.push({ pattern: "^" + loose + "$", type: "loose_type_match", score: 75 });
    }
  }

  if (positives.length >= 2) {
    const exactOr = positives.map(p => escapeRegex(p)).join("|");
    candidates.push({ pattern: "^(" + exactOr + ")$", type: "exact_or", score: 90 });
  }

  const substrPatterns = new Set();
  for (const p of positives) {
    for (let len = 3; len <= Math.min(p.length, 10); len++) {
      for (let i = 0; i <= p.length - len; i++) {
        const sub = p.slice(i, i + len);
        if (/^[a-zA-Z0-9]+$/.test(sub)) {
          if (positives.every(x => x.includes(sub))) {
            substrPatterns.add(escapeRegex(sub));
          }
        }
      }
    }
  }
  substrPatterns.forEach(pat => {
    candidates.push({ pattern: pat, type: "common_substring", score: 50 });
  });

  const uniquePatterns = [];
  const seen = new Set();
  for (const c of candidates) {
    if (!seen.has(c.pattern)) {
      seen.add(c.pattern);
      uniquePatterns.push(c);
    }
  }

  return uniquePatterns.sort((a, b) => b.score - a.score);
}

function filterCandidates(candidates, positives, negatives) {
  const results = [];

  for (const cand of candidates) {
    try {
      const regex = new RegExp(cand.pattern);

      const truePositives = positives.filter(p => regex.test(p)).length;
      const falsePositives = negatives.filter(n => regex.test(n)).length;

      const recall = positives.length > 0 ? truePositives / positives.length : 0;
      const precision = (truePositives + falsePositives) > 0 ? truePositives / (truePositives + falsePositives) : 1;
      const f1 = (recall + precision) > 0 ? 2 * recall * precision / (recall + precision) : 0;

      if (recall === 1 && falsePositives === 0) {
        cand.score = Math.min(100, cand.score + 20);
      } else if (recall === 1) {
        cand.score = Math.min(95, cand.score + 10);
      } else if (falsePositives === 0) {
        cand.score = Math.max(20, cand.score - 5);
      } else {
        cand.score = Math.max(10, cand.score - 20 - falsePositives * 10);
      }

      results.push({
        ...cand,
        recall,
        precision,
        f1,
        true_positives: truePositives,
        false_positives: falsePositives,
        matches_all_positives: recall === 1,
        rejects_all_negatives: falsePositives === 0,
      });
    } catch (e) {
    }
  }

  return results
    .filter(r => r.true_positives > 0)
    .sort((a, b) => b.f1 - a.f1 || b.score - a.score);
}

function generateRegexFromSamples(samples) {
  const { positives = [], negatives = [], flags = [], maxCandidates = 10 } = samples;

  if (positives.length === 0) {
    return { candidates: [], message: "请至少提供一个正例样本" };
  }

  let candidates = generateCandidates(positives);
  candidates = filterCandidates(candidates, positives, negatives);
  candidates = candidates.slice(0, maxCandidates);

  return {
    candidates,
    stats: {
      positive_count: positives.length,
      negative_count: negatives.length,
      candidates_generated: candidates.length,
    },
    tips: generateTips(candidates, positives, negatives),
  };
}

function generateTips(candidates, positives, negatives) {
  const tips = [];

  const perfect = candidates.find(c => c.matches_all_positives && c.rejects_all_negatives);
  if (perfect) {
    tips.push({ type: "success", message: "已找到完美匹配的正则表达式！" });
  } else {
    const matchAll = candidates.filter(c => c.matches_all_positives);
    if (matchAll.length === 0) {
      tips.push({ type: "warning", message: "未找到能匹配所有正例的正则，建议添加更多正例样本" });
    } else if (negatives.length > 0) {
      tips.push({ type: "info", message: "当前正则可能匹配部分反例，建议添加更多反例样本细化规则" });
    }
  }

  if (positives.length < 3) {
    tips.push({ type: "info", message: "添加更多正例样本可以生成更准确的正则" });
  }
  if (negatives.length === 0) {
    tips.push({ type: "info", message: "添加反例（不应匹配的字符串）可以帮助过滤掉过于宽泛的规则" });
  }

  return tips;
}

module.exports = { generateRegexFromSamples };
