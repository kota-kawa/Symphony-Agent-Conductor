import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";

const LONG_TERM_CATEGORIES = [
  "profile",
  "preference",
  "health",
  "work",
  "hobby",
  "relationship",
  "life",
  "travel",
  "food",
  "general",
];

const SHORT_TERM_CATEGORIES = [
  "active_task",
  "pending_questions",
  "recent_entities",
  "emotional_context",
  "general",
];

const CATEGORY_LABELS: Record<string, string> = {
  profile: "基本情報",
  preference: "好み・嗜好",
  health: "健康",
  work: "仕事・学業",
  hobby: "趣味",
  relationship: "人間関係",
  life: "生活",
  travel: "旅行",
  food: "食事",
  general: "その他・メモ",
  active_task: "現在進行中のタスク",
  pending_questions: "未解決の質問",
  recent_entities: "直近の話題・キーワード",
  emotional_context: "現在の感情・雰囲気",
};

const PLACEHOLDER: Record<string, string> = {
  profile: "例: 名前は山田太郎。東京在住。30代。エンジニアとして働いている。",
  preference: "例: 返答は簡潔が好き。敬体が好み。長文より箇条書きが助かる。",
  health: "例: 毎日朝にジョギング。カフェイン控えめを希望。",
  work: "例: プロジェクトXの締切は毎週金曜。リモート勤務中心。",
  hobby: "例: ロードバイクと写真が趣味。休日は多摩川沿いを走る。",
  relationship: "例: 佐藤さんとは同僚。田中さんはメンター。",
  life: "例: 早朝型。家事は週末にまとめて行う。",
  travel: "例: 夏に北海道旅行を計画中。温泉が好き。",
  food: "例: 和食とコーヒーが好き。辛すぎる料理は苦手。",
  general: "例: 雑多なメモや、まだ分類できていない情報。",
  active_task: "例: タスク: 旅行の計画を立てる (ステータス: 進行中)",
  pending_questions: "例: 質問: 次回の会議はいつ？\n質問: あのレストランの名前は？",
  recent_entities: "例: キーワード: React, Python, 温泉",
  emotional_context: "例: 気分: 落ち着いている。少し急ぎ。",
};

function formatShortTermValue(category: string, data: string | undefined, fullMemory: any) {
  if (category === "active_task") {
    const task = fullMemory.active_task || {};
    if (task.goal) {
      return `タスク: ${task.goal}\nステータス: ${task.status || "active"}`;
    }
  }

  if (category === "pending_questions") {
    const questions = fullMemory.pending_questions || [];
    if (Array.isArray(questions) && questions.length > 0) {
      return questions.map((q: string) => `質問: ${q}`).join("\n");
    }
  }

  if (category === "recent_entities") {
    const entities = fullMemory.recent_entities || [];
    if (Array.isArray(entities) && entities.length > 0) {
      const names = entities.map((e: any) => e.name).filter(Boolean);
      if (names.length > 0) {
        return `キーワード: ${names.join(", ")}`;
      }
    }
  }

  if (category === "emotional_context") {
    if (fullMemory.emotional_context) {
      return `気分: ${fullMemory.emotional_context}`;
    }
  }

  if (data && typeof data === "string") {
    return data;
  }

  return "";
}

type MemoryPayload = {
  long_term_categories?: Record<string, string>;
  short_term_categories?: Record<string, string>;
  long_term_full?: Record<string, any>;
  short_term_full?: Record<string, any>;
};

const MemoryApp: React.FC = () => {
  const [longData, setLongData] = useState<Record<string, string>>({});
  const [shortData, setShortData] = useState<Record<string, string>>({});
  const [longFull, setLongFull] = useState<Record<string, any>>({});
  const [shortFull, setShortFull] = useState<Record<string, any>>({});
  const [status, setStatus] = useState<string>("");

  useEffect(() => {
    fetch("/api/memory")
      .then((response) => response.json())
      .then((data: MemoryPayload) => {
        setLongData(data.long_term_categories || {});
        setShortData(data.short_term_categories || {});
        setLongFull(data.long_term_full || {});
        setShortFull(data.short_term_full || {});
      })
      .catch((error) => {
        console.error("Error fetching memory:", error);
        setStatus("メモリの読み込みに失敗しました。");
      });
  }, []);

  const longTermValues = useMemo(() => {
    const values: Record<string, string> = {};
    LONG_TERM_CATEGORIES.forEach((cat) => {
      values[cat] = longData[cat] || "";
    });
    return values;
  }, [longData]);

  const shortTermValues = useMemo(() => {
    const values: Record<string, string> = {};
    SHORT_TERM_CATEGORIES.forEach((cat) => {
      if (cat !== "general") {
        values[cat] = formatShortTermValue(cat, shortData[cat], shortFull);
      } else {
        values[cat] = shortData[cat] || "";
      }
    });
    return values;
  }, [shortData, shortFull]);

  const [longInputs, setLongInputs] = useState<Record<string, string>>({});
  const [shortInputs, setShortInputs] = useState<Record<string, string>>({});

  useEffect(() => {
    setLongInputs(longTermValues);
  }, [longTermValues]);

  useEffect(() => {
    setShortInputs(shortTermValues);
  }, [shortTermValues]);

  const handleSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    setStatus("保存中...");

    const longTermData: Record<string, string> = {};
    Object.entries(longInputs).forEach(([key, value]) => {
      const trimmed = value.trim();
      if (trimmed) {
        longTermData[key] = trimmed;
      }
    });
    const shortTermData: Record<string, string> = {};
    Object.entries(shortInputs).forEach(([key, value]) => {
      const trimmed = value.trim();
      if (trimmed) {
        shortTermData[key] = trimmed;
      }
    });

    fetch("/api/memory", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        long_term_memory: longTermData,
        short_term_memory: shortTermData,
      }),
    })
      .then((response) => response.json())
      .then((data) => {
        setStatus(data.message || "保存しました。");
        setTimeout(() => setStatus(""), 3000);
      })
      .catch((error) => {
        console.error("Error saving memory:", error);
        setStatus("メモリの保存に失敗しました。");
      });
  };

  return (
    <div className="memory-container">
      <h1>Memory Management</h1>
      <div className="subtitle">短期・長期記憶を「自然言語」で編集できます。JSONは不要です。</div>
      <div className="memory-note">
        推奨カテゴリは「プロフィール / 好み・嗜好 / 健康 / 仕事・学業 / 趣味 / 人間関係 / 生活 / 旅行 / 食事 / その他」です。
        JSONなどで直接編集する場合も、上記ラベルを使用するとUIとオーケストレーター双方で確実に参照されます。
        それ以外のカテゴリを入力した場合は、自動的に追加カテゴリとして扱われます。
      </div>
      <form id="memoryForm" className="memory-form" onSubmit={handleSubmit}>
        <div>
          <div className="section-title">長期記憶（永続・変わりにくい事実）</div>
          <div id="longTermSections" className="memory-grid">
            {LONG_TERM_CATEGORIES.map((cat) => (
              <div key={cat} className="memory-card">
                <label htmlFor={`longTerm-${cat}`}>{CATEGORY_LABELS[cat] || cat}</label>
                <div className="memory-hint">{PLACEHOLDER[cat] || ""}</div>
                <textarea
                  id={`longTerm-${cat}`}
                  data-category={cat}
                  value={longInputs[cat] || ""}
                  placeholder={PLACEHOLDER[cat] || ""}
                  onChange={(e) => setLongInputs((prev) => ({ ...prev, [cat]: e.target.value }))}
                />
              </div>
            ))}
          </div>
        </div>
        <div>
          <div className="section-title">短期記憶（直近のコンテキスト・保留事項）</div>
          <div id="shortTermSections" className="memory-grid">
            {SHORT_TERM_CATEGORIES.map((cat) => (
              <div key={cat} className="memory-card">
                <label htmlFor={`shortTerm-${cat}`}>{CATEGORY_LABELS[cat] || cat}</label>
                <div className="memory-hint">{PLACEHOLDER[cat] || ""}</div>
                <textarea
                  id={`shortTerm-${cat}`}
                  data-category={cat}
                  value={shortInputs[cat] || ""}
                  placeholder={PLACEHOLDER[cat] || ""}
                  onChange={(e) => setShortInputs((prev) => ({ ...prev, [cat]: e.target.value }))}
                />
              </div>
            ))}
          </div>
        </div>
        <div className="save-bar">
          <p id="statusMessage" className="status">{status}</p>
          <button type="submit" className="save-button">保存する</button>
        </div>
      </form>
    </div>
  );
};

const mount = document.getElementById("memory-root");
if (mount) {
  createRoot(mount).render(<MemoryApp />);
}
