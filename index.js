/**
 * Host half of the politics subjective trainer.
 *
 * The Client owns the question bank and local progress. This half makes one
 * rubric-bound model call per submission through `ctx.llm`.
 */
export const inject = ['llm', 'webServer']

const PREFIX = '/politics-subjective'
const BUILD = 'v1'
const REQUEST_TIMEOUT_MS = 180000

const SYSTEM_EVALUATOR = `你是一位考研政治阅卷组资深教师，熟悉考研政治主观题的评分标准与命题规律。
你的任务是严格、具体、可执行地批改学生的政治主观题作答，而不是泛泛鼓励。
评分维度：
- 理论观点是否准确、是否符合教材与考试大纲；
- 要点是否完整，是否覆盖设问要求的层次；
- 是否结合材料，而不是只背原理；
- 逻辑结构是否清楚，是否分点作答；
- 语言是否规范，术语使用是否准确。
你必须只输出一个 JSON 对象。不要输出 Markdown 代码围栏，不要输出 JSON 之外的任何解释文字。
参考答案要点要明确标注为“参考要点”，不要冒充官方标准答案；不确定的地方要说明。`

function json(res, status, body) {
  const text = JSON.stringify(body)
  res.writeHead(status, {
    'content-type': 'application/json; charset=utf-8',
    'cache-control': 'no-store',
    'content-length': Buffer.byteLength(text),
  })
  res.end(text)
}

async function readBody(req) {
  const chunks = []
  for await (const chunk of req) chunks.push(chunk)
  if (chunks.length === 0) return {}
  try {
    return JSON.parse(Buffer.concat(chunks).toString('utf8'))
  } catch {
    return {}
  }
}

function textOf(value, fallback = '') {
  return typeof value === 'string' && value.trim() ? value.trim() : fallback
}

function listOf(value) {
  return Array.isArray(value) ? value.filter((item) => typeof item === 'string' && item.trim()) : []
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value))
}

function roundHalf(value) {
  return Math.round(value * 2) / 2
}

async function pickModel(ctx) {
  try {
    const selection = ctx.get('agentDefaultModel')?.currentSelection?.()
    if (selection && selection.provider && selection.model) {
      return { provider: selection.provider, model: selection.model }
    }
  } catch {
    /* fall through */
  }
  const providers = ctx.llm.listProviders()
  for (const provider of providers) {
    try {
      const models = await ctx.llm.listModels(provider.id)
      if (models.length > 0) return { provider: provider.id, model: models[0].id }
    } catch {
      /* try the next route */
    }
  }
  if (providers.length > 0) return { provider: providers[0].id, model: 'deepseek-flash' }
  throw new Error('当前 Profile 没有可用的模型路由，请先在设置中登录或配置模型。')
}

async function effortFor(ctx, provider, model) {
  try {
    const info = await ctx.llm.resolveModelInfo(provider, model)
    const efforts = info && info.reasoning && Array.isArray(info.reasoning.efforts) ? info.reasoning.efforts : []
    if (!efforts.length) return undefined
    const low = efforts.find((effort) => effort.id === 'low')
    return low ? low.id : efforts[0].id
  } catch {
    return undefined
  }
}

function extractJson(raw) {
  let text = String(raw || '').trim()
  if (!text) return null
  text = text.replace(/^```(?:json)?\s*/i, '').replace(/```\s*$/i, '').trim()
  try {
    return JSON.parse(text)
  } catch {
    /* locate the outermost object below */
  }
  const start = text.indexOf('{')
  const end = text.lastIndexOf('}')
  if (start >= 0 && end > start) {
    try {
      return JSON.parse(text.slice(start, end + 1))
    } catch {
      /* give up */
    }
  }
  return null
}

async function complete(ctx, system, user, maxTokens) {
  const { provider, model } = await pickModel(ctx)
  const reasoningEffort = await effortFor(ctx, provider, model)
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS)
  let raw = ''
  let blockText = ''
  let failure = null
  let finishKind = ''
  let usage = null
  try {
    const stream = ctx.llm.stream({
      provider,
      model,
      system,
      messages: [{ role: 'user', content: [{ type: 'text', text: user }] }],
      temperature: 0.2,
      ...(reasoningEffort ? { reasoningEffort } : {}),
      maxTokens: Math.max(maxTokens, 6000),
      signal: controller.signal,
    })
    for await (const chunk of stream) {
      if (chunk.type === 'text-delta') {
        raw += chunk.text
      } else if (chunk.type === 'block-end') {
        if (chunk.block && chunk.block.type === 'text') blockText += chunk.block.text
      } else if (chunk.type === 'usage') {
        usage = chunk.usage
      } else if (chunk.type === 'finish') {
        finishKind = chunk.reason.kind
        if (chunk.reason.kind === 'error' || chunk.reason.kind === 'aborted') {
          failure = chunk.reason.failure
        }
      }
    }
  } catch (error) {
    if (controller.signal.aborted) throw new Error('模型请求超时，请稍后重试。')
    throw error
  } finally {
    clearTimeout(timer)
  }
  if (failure) {
    throw new Error(`${failure.code || 'LLM_ERROR'}${failure.message ? `：${failure.message}` : ''}`)
  }
  const text = raw || blockText
  if (!text.trim()) {
    const used = usage ? `，输出 ${usage.outputTokens || 0} tokens` : ''
    throw new Error(
      `模型返回了空内容（finish=${finishKind || 'none'}${used}）。请稍后重试，或在设置中换一个模型。`,
    )
  }
  return { raw: text, parsed: extractJson(text), provider, model }
}

function normalizeRubric(input, maxScore) {
  if (!Array.isArray(input)) return []
  return input
    .filter((row) => row && typeof row === 'object')
    .slice(0, 6)
    .map((row) => {
      const max = Number(row.max)
      const score = Number(row.score)
      return {
        criterion: textOf(row.criterion, '评分项'),
        score: Number.isFinite(score) ? clamp(score, 0, Number.isFinite(max) && max > 0 ? max : maxScore) : 0,
        max: Number.isFinite(max) && max > 0 ? max : Math.max(1, Math.round(maxScore / 4)),
        comment: textOf(row.comment),
      }
    })
}

function normalizePoints(input) {
  if (!Array.isArray(input)) return []
  return input
    .filter((row) => row && typeof row === 'object')
    .slice(0, 10)
    .map((row) => ({
      point: textOf(row.point || row.expression || row.name),
      explanation: textOf(row.explanation || row.why || row.meaning),
    }))
    .filter((row) => row.point)
}

function normalizeMissing(input) {
  if (!Array.isArray(input)) return []
  return input
    .filter((row) => row && typeof row === 'object')
    .slice(0, 10)
    .map((row) => ({
      point: textOf(row.point || row.name),
      why: textOf(row.why || row.explanation),
    }))
    .filter((row) => row.point)
}

function normalizeIssues(input) {
  if (!Array.isArray(input)) return []
  return input
    .filter((row) => row && typeof row === 'object')
    .slice(0, 10)
    .map((row) => ({
      quote: textOf(row.quote),
      problem: textOf(row.problem, '需要改进'),
      fix: textOf(row.fix, '按高分答案的结构重写这一处。'),
    }))
}

function normalizeReusable(input) {
  if (!Array.isArray(input)) return []
  return input
    .filter((row) => row && typeof row === 'object')
    .slice(0, 10)
    .map((row) => ({
      expression: textOf(row.expression),
      meaning: textOf(row.meaning),
      usage: textOf(row.usage),
    }))
    .filter((row) => row.expression)
}

function normalizePerQuestion(input) {
  if (!Array.isArray(input)) return []
  return input
    .filter((row) => row && typeof row === 'object')
    .slice(0, 10)
    .map((row) => ({
      number: Number(row.number) || 0,
      score: Number(row.score) || 0,
      max: Number(row.max) || 0,
      comment: textOf(row.comment),
      missingPoints: normalizeMissing(row.missingPoints),
    }))
}

function normalizeEvaluation(parsed, maxScore) {
  const rawScore = Number(parsed && parsed.score)
  const score = Number.isFinite(rawScore) ? roundHalf(clamp(rawScore, 0, maxScore)) : null
  return {
    score,
    maxScore,
    band: textOf(parsed && parsed.band),
    summary: textOf(parsed && parsed.summary, '已完成批改。'),
    rubric: normalizeRubric(parsed && parsed.rubric, maxScore),
    referencePoints: normalizePoints(parsed && parsed.referencePoints),
    missingPoints: normalizeMissing(parsed && parsed.missingPoints),
    issues: normalizeIssues(parsed && parsed.issues),
    revised: textOf(parsed && parsed.revised),
    reusable: normalizeReusable(parsed && parsed.reusable),
    perQuestion: normalizePerQuestion(parsed && parsed.perQuestion),
    nextStep: textOf(parsed && parsed.nextStep),
  }
}

function questionBrief(question) {
  if (!question || typeof question !== 'object') return '（题目信息缺失）'
  return [
    `${question.year} 年第 ${question.number} 题 · ${question.subject} · 满分 ${question.score} 分`,
    textOf(question.text),
  ].join('\n')
}

function singlePrompt(body) {
  const question = body.question || {}
  const maxScore = Number(question.score) || 10
  const text = textOf(body.text)
  return `请批改一道考研政治主观题。

【题目】${questionBrief(question)}

【学生答案】
<<<
${text}
>>>

满分 ${maxScore} 分。请按考研政治主观题阅卷标准批改：理论观点是否准确、要点是否完整、是否结合材料、逻辑结构是否清楚、术语与语言是否规范。
请给出参考要点（明确标注“参考要点”，不是官方答案）、学生遗漏的要点、具体问题与改法、以及一份高分答案提纲/示范。
只输出以下 JSON：
{
  "score": 0-${maxScore} 的数字,
  "band": "例如：第一档 / 第二档",
  "summary": "一句话总评",
  "rubric": [{"criterion":"理论观点","score":0,"max":0,"comment":"具体判断"}],
  "referencePoints": [{"point":"参考要点","explanation":"简要说明"}],
  "missingPoints": [{"point":"学生遗漏的要点","why":"为什么重要"}],
  "issues": [{"quote":"学生原句","problem":"问题","fix":"可直接替换/补充的写法"}],
  "revised": "高分答案提纲或示范",
  "reusable": [{"expression":"可复用术语/答题句式","meaning":"含义","usage":"适用场景"}],
  "nextStep": "下一题最该补的一个点"
}`
}

function setPrompt(body) {
  const set = body.set || {}
  const questions = Array.isArray(set.questions) ? set.questions : []
  const maxScore = questions.reduce((sum, question) => sum + (Number(question.score) || 0), 0)
  const text = textOf(body.text)
  const questionLines = questions
    .map((question) => `第 ${question.number} 题（${question.subject}，${question.score} 分）\n${question.text}`)
    .join('\n\n')
  return `请批改一套考研政治主观题套题。

【套题】${set.year || ''} 年，共 ${questions.length} 题，满分 ${maxScore} 分。
${questionLines}

【学生答案】
<<<
${text}
>>>

请按题号对应学生答案；若学生答案没有按题号分段，请根据内容判断归属并说明。逐题给出得分与主要问题，再给整套总评。
只输出以下 JSON：
{
  "score": 0-${maxScore} 的数字,
  "band": "例如：第一档 / 第二档",
  "summary": "整套总评",
  "rubric": [{"criterion":"理论观点","score":0,"max":0,"comment":"具体判断"}],
  "perQuestion": [{"number":34,"score":0,"max":10,"comment":"该题评价","missingPoints":[{"point":"遗漏要点","why":"原因"}]}],
  "referencePoints": [{"point":"整套参考要点","explanation":"简要说明"}],
  "missingPoints": [{"point":"整体遗漏","why":"原因"}],
  "issues": [{"quote":"学生原句","problem":"问题","fix":"改法"}],
  "revised": "整套高分答案提纲/示范",
  "reusable": [{"expression":"可复用术语/句式","meaning":"含义","usage":"适用场景"}],
  "nextStep": "下一套最该补的一个点"
}`
}

async function evaluate(ctx, body) {
  const mode = body.mode === 'set' ? 'set' : 'single'
  const text = textOf(body.text)
  if (text.length < 20) throw new Error('答案太短，至少写出一段完整表述再提交。')
  if (text.length > 20000) throw new Error('答案过长，请控制在 20000 字符以内。')
  const maxScore =
    mode === 'set'
      ? (Array.isArray(body.set && body.set.questions) ? body.set.questions : []).reduce(
          (sum, question) => sum + (Number(question.score) || 0),
          0,
        )
      : Number(body.question && body.question.score) || 10
  const prompt = mode === 'set' ? setPrompt(body) : singlePrompt(body)
  const result = await complete(ctx, SYSTEM_EVALUATOR, prompt, mode === 'set' ? 4200 : 3000)
  return {
    mode,
    maxScore,
    evaluation: result.parsed ? normalizeEvaluation(result.parsed, maxScore) : null,
    raw: result.parsed ? undefined : result.raw,
    model: `${result.provider}/${result.model}`,
  }
}

async function handle(req, res, ctx) {
  const url = new URL(req.url || '/', 'http://127.0.0.1')
  const route = url.pathname.slice(url.pathname.startsWith(PREFIX) ? PREFIX.length : 0) || '/'
  const method = (req.method || 'GET').toUpperCase()

  try {
    if (route === '/health' && method === 'GET') {
      return json(res, 200, { ok: true, build: BUILD, prefix: PREFIX })
    }
    if (route === '/evaluate' && method === 'POST') {
      const body = await readBody(req)
      const payload = await evaluate(ctx, body)
      return json(res, 200, { ok: true, ...payload })
    }
    return json(res, 404, { ok: false, error: `未知接口 ${method} ${route}` })
  } catch (error) {
    return json(res, 502, { ok: false, error: String((error && error.message) || error) })
  }
}

export function apply(ctx) {
  ctx.effect(
    () => ctx.webServer.register({ kind: 'prefix', path: PREFIX, handler: (req, res) => handle(req, res, ctx) }),
    'politics-subjective-route',
  )
}
