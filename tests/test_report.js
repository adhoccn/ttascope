/** 报告与文件解析单元测试（Node 直接跑，不需要浏览器）。
 *
 *   node tests/test_report.js
 *
 * 直接从 index.html 抽取被测函数，确保测的就是页面实际运行的代码。
 */
const fs = require('fs');
const path = require('path');
const src = fs.readFileSync(path.join(__dirname, '..', 'app', 'web', 'index.html'), 'utf-8');
const js = src.match(/<script>([\s\S]*)<\/script>/)[1];

// 桩：报告里只读 $('#compare').textContent
global.$ = () => ({ textContent: '交叉对照（15D 手工特征）：AI 块 9/24' });

const pick = ['function stamp', 'function splitCsvRow', 'function csvPick', 'function parseCsv',
              'function parseJsonl', 'function mdCell', 'function csvCell',
              'function buildReport', 'function buildCsv'];
for (const head of pick) {
  const a = js.indexOf(head);
  if (a < 0) throw new Error('未找到 ' + head);
  // 取到下一个顶层 function / 空行后的顶格符号
  const rest = js.slice(a);
  const end = rest.indexOf('\n}');            // 顶格大括号即函数结束
  if (end < 0) throw new Error('未找到函数结尾 ' + head);
  eval(rest.slice(0, end + 2));
}
const SPLIT_LABEL = { line: 'A', para: 'B', sent: 'C', chunk: 'D', whole: 'E' };
const FEAT_LABEL = { char_2_4: '字符 2-4gram TF-IDF' };
let SRCFILES = [];

// --- CSV ---
const csv = 'id,text,label\n1,"人工智能正在改变，各行各业的生产方式。",1\n2,"我跟你说啊，这事儿我也说不好。",0\n';
const c1 = parseCsv(csv);
console.log('CSV 取表头列名 ->', JSON.stringify(c1));
console.assert(c1.length === 2 && c1[0].startsWith('人工智能'), 'CSV text 列识别失败');

const csv2 = 'a,b,c\n111,短列,这是一段明显更长的小说一样的文本内容，用来试探最长列策略。\n222,短列,另一段同样较长的内容文本 placeholder.\n';
const c2 = parseCsv(csv2);
console.log('CSV 无表头取最长列 ->', JSON.stringify(c2));
console.assert(c2.length === 2 && c2[0].includes('小说'), 'CSV 最长列策略失败');

// 含引号转义与换行字段
const csv3 = 'text\n"他说：""今天天气不错""，然后就走了。"\n';
console.log('CSV 引号转义 ->', JSON.stringify(parseCsv(csv3)));

// --- JSONL ---
const jl = '{"id":1,"text":"第一段内容"}\n{"content":"第二段内容"}\n不是JSON的行\n';
console.log('JSONL ->', JSON.stringify(parseJsonl(jl)));
console.assert(parseJsonl(jl).length === 3, 'JSONL 解析失败');

// --- Markdown / CSV 单元格转义 ---
console.log('md 转义 ->', mdCell('a|b\n c'));
console.log('csv 转义 ->', csvCell('a"b,c'));

// --- 报告 ---
LAST = {
  d: { n: 3, chosen: 'bn', predicted_f1: 0.91, ai_rate: 2 / 3, elapsed_ms: 123,
       candidates: { none: 0.80, bn: 0.91 }, warn: '批次仅 3 条：诊断量仅供参考。',
       diag: { Xi_ai: 1.2, Xi_hu: -0.1, Delta_T: 2.0, Delta: 5.0, SNR_T: 0.5,
               pred_f1: 0.9, pred_f1_source: 0.95, expected_gain: 0.05,
               identity_resid: 0, band: '>1.2（11 组）' },
       items: [{ index: 0, p_ai: 0.91, label: 'AI' },
               { index: 1, p_ai: 0.70, label: 'AI' },
               { index: 2, p_ai: 0.30, label: '人类' }] },
  texts: ['第一段，人工智能相关的内容。', '第二段，同样是机器风格的行文。', '第三段，口语化的随意表达。'],
  feat: 'char_2_4', mode: 'auto', how: 'sent', size: '60'
};
const md = buildReport(LAST.d.items);
console.log('\n===== 报告预览（前 46 行）=====');
console.log(md.split('\n').slice(44).join('\n'));
console.log('\n===== 报告检查 =====');
console.assert(md.includes('## 四、逐段分析'), '缺逐段分析章节');
console.assert(md.includes('## 五、逐段原文'), '缺逐段原文章节');
console.assert(/\| 1 \|/.test(md) && /\| 2 \|/.test(md) && /\| 3 \|/.test(md), '逐段表格行数不全');
console.assert(md.includes('第三段，口语化的随意表达。'), '逐段原文缺全文');
// 逐段表格不能被原文里的换行折断：区段内每行列数必须与表头一致
const seg = md.split('## 四、逐段分析')[1].split('## 五、')[0].split('\n')
              .filter(l => l.trim().startsWith('|'));
const ncol = l => l.split(/(?<!\\)\|/).length;
const headCols = ncol(seg[0]);
const badRows = seg.filter(l => ncol(l) !== headCols);
console.assert(badRows.length === 0, '逐段表格列数不一致：' + badRows.join(' ⏎ '));
console.assert(seg.length === 1 + 1 + LAST.texts.length,
               '逐段表格行数应为 表头+分隔+数据，实为 ' + seg.length);
console.log('报告字符数', md.length);

console.log('\n===== CSV =====');
console.log(buildCsv(LAST.d.items));
console.log('\n全部断言通过');
