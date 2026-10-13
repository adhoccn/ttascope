/** 无头 Edge 实测页面交互（需 puppeteer-core）。
 *
 *   NODE_PATH=<workspace>/node_modules node tests/browser_check.js \
 *       http://106.54.192.87:8501/tta/
 *
 * 覆盖：空输入禁用 → 不足门槛禁用并提示还差几条 → 示例达标后可检测并渲染
 * → 长文按句切块后整篇结论出现。任一环节异常即退出码 1。
 */
const puppeteer = require('puppeteer-core');
const fs = require('fs');
const os = require('os');
const path = require('path');

const EDGE = 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';
const URL = process.argv[2] || 'http://106.54.192.87:8501/tta/';
const dlDir = path.join(os.tmpdir(), 'ttascope-dl-' + Date.now());
let fail = 0;

function check(name, cond, extra) {
  console.log((cond ? '  ok  ' : ' FAIL ') + name + (extra ? ' | ' + extra : ''));
  if (!cond) fail++;
}

(async () => {
  const browser = await puppeteer.launch({
    executablePath: EDGE, headless: 'new',
    args: ['--no-sandbox', '--disable-dev-shm-usage']
  });
  const page = await browser.newPage();
  const errs = [];
  page.on('pageerror', e => errs.push('pageerror: ' + e.message));
  page.on('console', m => { if (m.type() === 'error') errs.push('console: ' + m.text()); });

  await page.goto(URL, { waitUntil: 'networkidle2', timeout: 60000 });
  const state = () => page.evaluate(() => ({
    n: document.querySelector('#count').textContent,
    disabled: document.querySelector('#run').disabled,
    cls: document.querySelector('#count').className
  }));

  // 1. 空输入
  let s = await state();
  check('空输入时按钮禁用', s.disabled === true, s.n);

  // 2. 不足门槛
  await page.$eval('#texts', el => {
    el.value = Array.from({length: 10}, (_, i) =>
      '测试文本第 ' + (i + 1) + ' 条，用于验证数量门槛是否生效。').join('\n');
    el.dispatchEvent(new Event('input'));
  });
  s = await state();
  check('10 条时按钮禁用', s.disabled === true, s.n);
  check('提示还差几条', /还差\s*10\s*条/.test(s.n), s.n);
  check('提示为红色警示', s.cls.includes('bad'), s.cls);

  const setFeat = async v => {
    await page.select('#feat', v);
    await new Promise(r => setTimeout(r, 150));
  };

  // 2b. 门槛随特征集变化：24 条在 char 下可用，切到 15D 应立即禁用
  await page.click('#demo');
  s = await state();
  check('示例 24 条在 char_2_4 下可检测', s.disabled === false, s.n);
  await setFeat('15D');
  s = await state();
  check('切到 15D 后同批被禁用（门槛 50）', s.disabled === true, s.n);
  check('提示的特征集门槛为 15D', /不足\s*50\s*条/.test(s.n), s.n);
  await setFeat('char_2_4');
  s = await state();
  check('切回 char_2_4 后恢复可用', s.disabled === false, s.n);

  // 3. 载入示例（24 条）
  await page.click('#demo');
  s = await state();
  check('示例 24 条时按钮可用', s.disabled === false, s.n);
  await page.click('#run');
  await page.waitForFunction(
    () => !document.querySelector('#itemcard').hidden ||
          document.querySelector('#err').innerHTML.includes('检测失败'),
    { timeout: 90000 });
  const shown = await page.$eval('#summary', el => !el.hidden);
  const kpis = await page.$$eval('#kpis .kpi', els =>
    els.map(e => e.querySelector('.k').textContent + '=' + e.querySelector('.v').textContent));
  check('检测结果渲染', shown, kpis.join('  '));

  // 4. 长文：双击载入示例 → 按句成块
  await page.$eval('#demo', el => el.dispatchEvent(new MouseEvent('dblclick', {bubbles: true})));
  s = await state();
  check('长文切块达标', s.disabled === false, s.n);
  await page.click('#run');
  await page.waitForFunction(
    () => !document.querySelector('#itemcard').hidden, { timeout: 90000 });
  const docShown = await page.$eval('#doccard', el => !el.hidden);
  const verdict = await page.$eval('#verdict', el => el.textContent.trim());
  check('整篇结论出现', docShown, verdict);
  const defFeat = await page.$eval('#feat', el => el.value);
  check('默认特征集为 char_2_4', defFeat === 'char_2_4', defFeat);
  await page.waitForFunction(
    () => /整篇判为|未做交叉对照|失败/.test(
      document.querySelector('#compare').textContent),
    { timeout: 60000 });
  const cmp = await page.$eval('#compare', el => el.textContent.trim());
  // 24 块不足 15D 的 50 条门槛，对照应被主动跳过并说明，而不是拿失效结果"验证"
  check('对照在不达标时被跳过并说明原因', /未做交叉对照/.test(cmp), cmp);

  // 5. 上传文件：csv 走列识别，txt 走按行
  const dir = path.join(__dirname, 'fixtures');
  fs.mkdirSync(dlDir, {recursive: true});
  const cdp = await page.target().createCDPSession();
  await cdp.send('Page.setDownloadBehavior',
                 {behavior: 'allow', downloadPath: dlDir});

  await page.$eval('#texts', el => { el.value = ''; el.dispatchEvent(new Event('input')); });
  await page.$eval('#split', el => {
    el.value = 'line'; el.dispatchEvent(new Event('change'));
  });
  let inp = await page.$('#file');
  await inp.uploadFile(path.join(dir, 'sample.csv'));
  await page.waitForFunction(
    () => /sample\.csv/.test(document.querySelector('#filenames').textContent),
    {timeout: 20000});
  s = await state();
  check('上传 CSV 后识别为 24 条', /^24 条/.test(s.n), s.n);
  const firstLine = await page.$eval('#texts', el => el.value.split('\n')[0]);
  check('CSV 取的是 text 列而非 id 列', /^人工智能/.test(firstLine), firstLine.slice(0, 24));

  inp = await page.$('#file');
  await inp.uploadFile(path.join(dir, 'sample.jsonl'));
  await page.waitForFunction(
    () => /sample\.jsonl/.test(document.querySelector('#filenames').textContent),
    {timeout: 20000});
  s = await state();
  check('追加 JSONL 后合计 48 条', /^48 条/.test(s.n), s.n);

  // 6. 跑检测并导出报告
  await page.click('#run');
  await page.waitForFunction(() => !document.querySelector('#itemcard').hidden,
                             {timeout: 90000});
  check('导出卡片出现', await page.$eval('#exportcard', el => !el.hidden));

  const grab = async btn => {
    await page.click(btn);
    await page.waitForFunction(
      () => /已导出|导出失败/.test(document.querySelector('#dlhint').textContent),
      {timeout: 120000});
    return page.$eval('#dlhint', el => el.textContent);
  };
  // Blob 下载是异步的，轮询等待文件出现且大小稳定后再读
  const waitFile = async (ext, ms = 20000) => {
    const t0 = Date.now(); let hit = null, size = -1;
    while (Date.now() - t0 < ms) {
      hit = fs.readdirSync(dlDir).find(f => f.endsWith(ext)) || null;
      if (hit) {
        const cur = fs.statSync(path.join(dlDir, hit)).size;
        if (cur > 0 && cur === size) return hit;
        size = cur;
      }
      await new Promise(r => setTimeout(r, 300));
    }
    return hit;
  };

  const hintMd = await grab('#dlmd');
  check('Markdown 报告导出成功', /已导出 ttascope-report-.*\.md/.test(hintMd), hintMd);

  const mdFile = await waitFile('.md');
  check('报告文件落盘', !!mdFile, fs.readdirSync(dlDir).join(', ') || '(空)');
  if (mdFile) {
    const md = fs.readFileSync(path.join(dlDir, mdFile), 'utf-8');
    check('报告含逐段分析章节', md.includes('## 四、逐段分析'));
    check('报告含逐段原文章节', md.includes('## 五、逐段原文'));
    const bodyRows = md.split('## 四、逐段分析')[1].split('## 五、')[0]
      .split('\n').filter(l => /^\| \d+ \|/.test(l));
    check('逐段表格覆盖全部 48 段', bodyRows.length === 48, '实为 ' + bodyRows.length);
    check('报告含 CSV 来源溯源', md.includes('sample.csv') && md.includes('sample.jsonl'));
    check('报告含漂移诊断量', /Ξ_ai/.test(md) && /Δ_T/.test(md));
    check('报告含适用边界', md.includes('## 六、适用边界'));
  }

  const hintCsv = await grab('#dlcsv');
  check('CSV 导出成功', /已导出 ttascope-report-.*\.csv/.test(hintCsv), hintCsv);
  const csvFile = await waitFile('.csv');
  if (csvFile) {
    const rows = fs.readFileSync(path.join(dlDir, csvFile), 'utf-8')
      .replace(/^\uFEFF/, '').trim().split('\r\n');
    check('CSV 行数 = 表头 + 48', rows.length === 49, '实为 ' + rows.length);
    check('CSV 首列为序号', rows[1].startsWith('"1"'), rows[1].slice(0, 30));
  }

  check('无 JS 报错', errs.length === 0, errs.join(' / '));
  await browser.close();
  fs.rmSync(dlDir, {recursive: true, force: true});
  console.log(fail ? '\n结果: FAIL（' + fail + ' 项）' : '\n结果: PASS');
  process.exit(fail ? 1 : 0);
})().catch(e => { console.error('FAIL', e.message); process.exit(1); });
