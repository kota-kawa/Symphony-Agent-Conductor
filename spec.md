# Project Specifications & Guidelines

## 目的・ゴール
- 現在のフロントエンドを React + TypeScript に移行し、挙動とデザインを一切変更しない。
- 既存の FastAPI ルーティング、API コントラクト、テンプレート変数（`browser_embed_url` など）を変更しない。
- UI 文言、エラーメッセージ、確認ダイアログ文言は完全一致を維持する。

## スコープ
- メイン SPA: `templates/index.html` と `assets/app.js` / `frontend/src/spa/*` / `assets/styles.css` に相当する UI。
- メモリ編集ページ: `templates/memory.html` と `assets/memory.js` に相当する UI。
- Scheduler UI: `templates/scheduler_*.html` と `assets/scheduler/scheduler.js` / `assets/scheduler/style.css` に相当する UI。
- アイコン資産 `assets/icons/` と外部 CDN（Google Fonts、Bootstrap Icons、Bootstrap）を現状のまま利用する。

## 変更禁止事項
- 既存の DOM 構造で参照されている `id`/`class`/`data-*` 名称は互換性を維持する。
- `window.BROWSER_EMBED_URL` などの `window` グローバル値、`meta` タグ名、クエリパラメータ名を変更しない。
- エンドポイント、HTTP メソッド、レスポンスの扱いを変更しない。
- 既存の CSS 見た目（色、余白、フォント、アニメーション）を変更しない。
- `chat_history.json` / `short_term_memory.json` / `long_term_memory.json` の更新条件を変えない。

## React + TypeScript 移行方針
- React は関数コンポーネント + Hooks、TypeScript は `strict` 前提で型を付与する。
- 既存の `assets/styles.css` と `assets/scheduler/style.css` をそのまま読み込み、クラス名を保持する。
- ビルド成果物は FastAPI が静的配信できる場所に出力し、`templates/*.html` から参照できること。
- `templates/index.html` / `templates/memory.html` / `templates/scheduler_*.html` のルーティングを維持し、クライアントルーティングは追加しない。

## UI/UX 仕様（メイン SPA）
- 画面構成は `#app` 直下の `.layout`、`.sidebar`、`.content` の3要素構成を維持する。
- ビュー切替は `data-view` と `#view-*` の対応を維持し、`active` クラスの付け替えで制御する。
- `#appTitle` の表示は `一般ビュー` / `リモートブラウザ` / `IoT ダッシュボード` / `Life` / `Schedule` を維持する。
- サイドバーのチャットタイトルとアイコンはビューに応じて切り替える。
- `sidebar-toggle` は CSS 変数 `--sidebar-toggle-top` を更新し、`layout.sidebar-collapsed` のトグルで折りたたむ。
- `sidebar-toggle` の `aria-*` 属性（`aria-label`/`aria-expanded`）を現状と同じタイミングで更新する。

## 一般ビュー（Proxy 表示）
- `#generalDefaultContent` の表示/非表示はオーケストレーターのエージェント選択により切替する。
- `#generalProxyStatus` の文言は `resolveAgentLabel` に基づく現行文言を維持する。
- `#generalProxyContainer` に iframe または noVNC ブラウザを表示する。

## Agent Result iframe
- `.agent-result-view[data-agent]` に `iframe` を動的生成し、`loading="lazy"` と `allow="fullscreen"` を維持する。
- `buildAgentResultUrl` は `browser`/`lifestyle`/`iot`/`scheduler` のベース URL とパスを現行ルールで解決する。
- `browser_agent_base` / `lifestyle_agent_base` / `iot_agent_base` / `scheduler_agent_base` のクエリパラメータと `meta` タグ値の優先順位を維持する。
- `AGENT_TO_VIEW_MAP` / `AGENT_RESULT_TARGETS` / `GENERAL_PROXY_AGENT_LABELS` のエイリアスを維持する。

## Browser 埋め込み（noVNC 互換）
- `browser_embed_url` クエリ、`window.BROWSER_EMBED_URL`、`meta[name='browser-embed-url']` の順で解決する。
- `autoconnect=1` `resize=scale` `scale=auto` `view_clip=false` を必ず保持する。
- `novnc.viewport.sync` の `postMessage` ペイロードと `novnc.viewport.request*` / `novnc.viewport.reload` のハンドリングを現状と一致させる。
- 接続失敗時のメッセージ内容、リトライ回数、遅延を維持する。
- Fullscreen ボタンの挙動と `aria-label` を維持する。

## 共通チャット（サイドバー）
- `#sidebarChatLog` にメッセージをレンダリングし、追加後に最下部へスクロールする。
- 送信フォームは `#sidebarChatForm`、入力欄は `#sidebarChatInput` を維持する。
- `#sidebarPauseBtn` と `#sidebarResetBtn` の活性/非活性とアイコン切替は `currentChatMode` によって現行と同一にする。

## メッセージ表示
- `createMessageElement` の構造、アイコン、時刻表示形式を維持する。
- `pending` メッセージは「AIが考えています / 見つけた情報から回答を組み立て中」を維持する。

## Life-Style（一般チャット）
- `GET /conversation_history` `POST /rag_answer` `GET /conversation_summary` `POST /reset_history` を同一の方法で利用する。
- 会話履歴の読み込み失敗時の表示文言とフォールバックを維持する。
- `lifestyle_base` クエリ、`window.LIFESTYLE_API_BASE`、`meta[name='lifestyle-api-base']` の解決順序を維持する。

## オーケストレーター
- `/orchestrator/chat` の SSE 解析とイベント名（`plan`/`before_execution`/`execution_progress`/`after_execution`/`complete`/`error`）を維持する。
- `/chat_history` を 3 秒間隔でポーリングし、差分がない場合は再描画しない。
- `prefixOrchestratorText` と `normalizeAssistantText` による重複排除を維持する。
- Browser Agent 実行時は `generalProxyAgent` と Browser mirror の状態遷移を維持する。
- `reset_chat_history` とブラウザエージェント一時停止の連動を維持する。

## Browser Agent（チャット + ストリーム）
- EventSource で `/api/stream` を購読し、`message`/`update`/`reset`/`status` を処理する。
- `/api/history` `/api/chat` `/api/reset` `/api/pause` `/api/resume` の挙動を維持する。
- 送信キュー、`agentRunning`、`paused` 状態の遷移を維持する。
- `[browser-agent-final]` マーカー検出時の general proxy 終了処理を維持する。

## IoT Dashboard
- `/api/devices` を 6 秒間隔でポーリングし、カード表示と空状態表示を維持する。
- デバイス登録・名称変更・削除のフローと確認ダイアログ文言を維持する。
- `iot_agent_base` の解決順序を維持する。

## IoT Chat
- `IOT_CHAT_GREETING` と送受信フローを維持する。
- `paused` 状態では送信を抑止する。

## Scheduler（サイドバー）
- `GET /api/chat/history` `POST /api/chat` `DELETE /api/chat/history` を維持する。
- `SCHEDULER_CHAT_GREETING` と送受信フローを維持する。

## Scheduler UI（専用ページ）
- モデル選択は `/scheduler_agent/api/models` と `/scheduler_agent/model_settings` を使用し、デフォルトモデルのフォールバックを維持する。
- チャットは `/scheduler_agent/api/chat` と `/scheduler_agent/api/chat/history` を使用し、送信/一時停止/リセットの挙動を維持する。
- `refreshView` による部分更新（`/scheduler-ui/calendar_partial`、`/scheduler-ui/day/:date/timeline`、`/scheduler-ui/day/:date/log_partial`）を維持する。
- `flash-highlight` のハイライト挙動を維持する。

## 設定ダイアログ
- `GET /api/memory` / `POST /api/memory` のデータ構造を維持する。
- メモリカテゴリ、プレースホルダ、短期メモリの整形ロジックを維持する。
- エージェント接続トグルは `/api/agent_connections` を使用する。
- モデル選択は `/api/model_settings` を使用し、エージェント未接続時の無効化を維持する。
- 取得時のタイムアウト（15 秒/5 秒）と保存時のステータスメッセージ表示を維持する。

## メモリ編集ページ
- `LONG_TERM_CATEGORIES` / `SHORT_TERM_CATEGORIES` とプレースホルダ文言を維持する。
- `/api/memory` から読み込み、`long_term_memory` / `short_term_memory` を POST する。

## エージェント接続ステータス
- `/api/agent_status` の取得タイミング（初回 + 30 秒間隔）を維持する。
- トップの `#agentStatusBanner` は非表示維持、設定ダイアログ内でのみ警告表示する。

## 全体ルール (General Rules)
- UI 文言・エラー文言は完全一致を維持する。
- API コントラクトやタイミングを変えない。
- 追加のトラッキングや外部通信を行わない。

## コーディング規約 (Coding Conventions)
- **TypeScript/React**: `strict`、関数コンポーネント、Hooks、`useEffect` の依存配列を厳守する。
- **CSS**: 既存の `assets/styles.css` と `assets/scheduler/style.css` を優先し、新規 CSS は最小限にする。
- **Lint/Format**: ESLint + Prettier を導入し、既存コードと整合する設定にする。

## 命名規則 (Naming Conventions)
- **Components**: `PascalCase`。
- **Hooks**: `useXxx`。
- **Functions/Variables**: `camelCase`。
- **Constants**: `SCREAMING_SNAKE_CASE`。
- **Files**: `kebab-case.tsx`、`kebab-case.ts`。

## ディレクトリ構成方針 (Directory Structure Policy)
- React ソースは `frontend/`（例: `frontend/src`）配下に集約し、出力先は FastAPI が配信できる `assets/` 配下にする。
- 共通ユーティリティは `frontend/src/utils`、API クライアントは `frontend/src/services`、型定義は `frontend/src/types` に分離する。

## エラーハンドリング方針 (Error Handling Policy)
- UI に表示するエラー文言は現状と完全一致にする。
- ネットワーク失敗時は現行と同じフォールバックメッセージを出す。
- エラー時でも UI が操作不能にならないように状態を復旧する。

## テスト方針 (Testing Policy)
- **Unit Tests**: SSE 解析、URL 解決、チャット整形、状態遷移の純粋関数を対象にテストする。
- **E2E Tests**: 主要フロー（ビュー切替、チャット送信、設定保存、IoT 操作、Scheduler 部分更新）を最低限カバーする。
