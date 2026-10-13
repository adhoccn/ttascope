/** 生成一份真实报告样例（供阅读/评审用），不参与 CI。
 *
 *   NODE_PATH=<workspace>/node_modules node tests/gen_sample_report.js \
 *       http://106.54.192.87:8501/tta/  out.md
 */
const puppeteer = require('puppeteer-core');
const fs = require('fs');
const os = require('os');
const path = require('path');

const EDGE = 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';
const URL = process.argv[2] || 'http://106.54.192.87:8501/tta/';
const OUT = process.argv[3] || path.join(__dirname, '..', 'sample_report.md');
const TXT = path.join(__dirname, 'fixtures', 'sample.txt');

(async () => {
  const dlDir = path.join(os.tmpdir(), 'ttascope-gen-' + Date.now());
  fs.mkdirSync(dlDir, {recursive: true});
  const browser = await puppeteer.launch({
    executablePath: EDGE, headless: 'new',
    args: ['--no-sandbox', '--disable-dev-shm-usage']
  });
  const page = await browser.newPage();
  await page.setViewport({width: 1280, height: 1600});
  await page.goto(URL, {waitUntil: 'networkidle2', timeout: 60000});
  const cdp = await page.target().createCDPSession();
  await cdp.send('Page.setDownloadBehavior',
                 {behavior: 'allow', downloadPath: dlDir});

  await page.$eval('#split', el => {
    el.value = 'line'; el.dispatchEvent(new Event('change'));
  });
  const inp = await page.$('#file');
  await inp.uploadFile(TXT);
  await page.waitForFunction(
    () => /sample\.txt/.test(document.querySelector('#filenames').textContent),
    {timeout: 20000});
  await page.click('#run');
  // 只等结果表；cross-check（#compare）仅在长文模式下产生，不能作为完成条件
  await page.waitForFunction(() => !document.querySelector('#itemcard').hidden,
                             {timeout: 120000});
  await page.click('#dlmd');
  await page.waitForFunction(
    () => /已导出/.test(document.querySelector('#dlhint').textContent),
    {timeout: 120000});

  let file = null;
  const t0 = Date.now();
  while (!file && Date.now() - t0 < 45000) {
    const all = fs.readdirSync(dlDir);
    file = all.find(f => f.endsWith('.md')) || null;
    if (file && fs.statSync(path.join(dlDir, file)).size === 0) file = null;
    if (!file) await new Promise(r => setTimeout(r, 500));
  }
  if (!file) throw new Error('报告未落盘，目录内容：' + fs.readdirSync(dlDir).join(','));
  fs.copyFileSync(path.join(dlDir, file), OUT);
  await browser.close();
  fs.rmSync(dlDir, {recursive: true, force: true});
  console.log('已写出 ' + OUT + '  ' +
              fs.statSync(OUT).size + ' 字节');
})().catch(e => { console.error('FAIL', e.message); process.exit(1); });
