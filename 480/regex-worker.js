const { parentPort, workerData } = require("worker_threads");

const { pattern, testString, flagStr } = workerData;

const FLAG_MAP = {
  IGNORECASE: "i",
  MULTILINE: "m",
  DOTALL: "s",
  VERBOSE: "x",
  UNICODE: "u",
  STICKY: "y",
  GLOBAL: "g",
};

try {
  const regex = new RegExp(pattern, flagStr);
  const matches = [];
  let match;

  while ((match = regex.exec(testString)) !== null) {
    const groups = [];
    if (match.length > 1) {
      for (let i = 1; i < match.length; i++) {
        groups.push({
          index: i,
          value: match[i] !== undefined ? match[i] : null,
          start: match.indices ? match.indices[i]?.[0] ?? null : null,
          end: match.indices ? match.indices[i]?.[1] ?? null : null,
        });
      }
    }

    const namedGroups = {};
    if (match.groups) {
      for (const [name, value] of Object.entries(match.groups)) {
        namedGroups[name] = {
          value: value !== undefined ? value : null,
          start: match.indices?.groups?.[name]?.[0] ?? null,
          end: match.indices?.groups?.[name]?.[1] ?? null,
        };
      }
    }

    matches.push({
      match: match[0],
      start: match.index,
      end: match.index + match[0].length,
      groups,
      named_groups: namedGroups,
    });

    if (!flagStr.includes("g")) break;
    if (match[0].length === 0) regex.lastIndex++;
  }

  parentPort.postMessage({ success: true, matches });
} catch (e) {
  parentPort.postMessage({ success: false, error: e.message });
}
