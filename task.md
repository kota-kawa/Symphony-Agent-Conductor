# Current Task Context

## 今回やること・目的 (Goal/Objective)
- 現在のフロントエンドを React + TypeScript に移行し、挙動・デザイン・文言を完全一致で維持する。

## やること (Must)
- [ ] 現状 UI / API / セレクタ / 文言の一覧化と固定化を行う。
- [ ] React + TypeScript のビルド環境を構築し、FastAPI テンプレートから静的に参照できる出力経路を決定する。
- [ ] SPA 構造を React で再構築し、`templates/index.html` に対応する DOM を生成する。
- [ ] レイアウト/ビュー切替、サイドバー折りたたみ、general proxy/noVNC のロジックを React に移植する。
- [ ] 共通チャット、オーケストレーター SSE、Browser Agent ストリーム、IoT/Scheduler チャットを移植する。
- [ ] 設定ダイアログ（メモリ/接続/モデル）と IoT 登録ダイアログを移植する。
- [ ] メモリ編集ページを React + TypeScript で再実装し、`/api/memory` の入出力を維持する。
- [ ] Scheduler UI ページ群のチャット・モデル選択・部分更新を React + TypeScript に移植する。
- [ ] 既存 CSS/アイコン/外部 CDN の参照を保持し、デザイン差分が出ないことを確認する。
- [ ] フロントエンドのユニットテストと最低限のE2E/手動検証手順を整備する。

## やらないこと (Non-goals)
- [ ] バックエンド API の変更や追加。
- [ ] UI 文言・色・レイアウト・タイポグラフィの変更。
- [ ] 既存のエージェント連携ロジックの仕様変更。
- [ ] ランタイム JSON の永続化方式の変更。

## 受け入れ基準 (Acceptance Criteria)
- [ ] `/` の SPA が現状と同一のレイアウトと挙動で動作する。
- [ ] ビュー切替、サイドバー折りたたみ、noVNC 埋め込み、各 iframe が現状と同一に動作する。
- [ ] 共通チャット、オーケストレーター SSE、Browser Agent/IoT/Scheduler チャットが現状と同一の文言とタイミングで表示される。
- [ ] 設定ダイアログの読み込み/保存、モデル選択、メモリ編集が現状と同一に機能する。
- [ ] `templates/memory.html` 相当の画面が同じ見た目と機能を維持する。
- [ ] Scheduler UI（カレンダー/日次/ルーチン画面）のチャット、モデル選択、部分更新が現状と同一に機能する。
- [ ] 既存の API エンドポイントの呼び出し回数・順序・パラメータが変わらない。
- [ ] 重大なコンソールエラーが発生しない。

## 影響範囲 (Impact/Scope)
- 触るファイル: `templates/index.html`
- 触るファイル: `templates/memory.html`
- 触るファイル: `templates/scheduler_*.html`
- 触るファイル: `assets/app.js`
- 触るファイル: `frontend/src/spa/*`
- 触るファイル: `assets/memory.js`
- 触るファイル: `assets/scheduler/scheduler.js`
- 触るファイル: `assets/styles.css`
- 触るファイル: `assets/scheduler/style.css`
- 触るファイル: `requirements.txt` またはフロントエンド用の新規設定ファイル
- 壊しちゃいけない挙動: ビュー切替、サイドバー、noVNC、オーケストレーター SSE、Browser Agent ストリーム、IoT 操作、Scheduler 部分更新。
