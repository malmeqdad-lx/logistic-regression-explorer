"""
Multinomial Logistic Regression Explorer
=========================================
An interactive Streamlit tool for visualizing and experimenting with
logistic regression for NLP intent classification.

Based on Jurafsky & Martin, Speech and Language Processing, Ch. 4
(Sections 4.1 – 4.8)

All algorithms implemented from scratch in NumPy — no sklearn/pytorch.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from collections import Counter
import re

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Logistic Regression Explorer · J&M Ch. 4",
    page_icon="📘",
    layout="wide",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Clean typography */
    .main .block-container { max-width: 1200px; padding-top: 1.5rem; }
    h1 { color: #1a1a2e; }
    h2 { color: #16213e; border-bottom: 2px solid #e94560; padding-bottom: 0.3rem; }
    h3 { color: #0f3460; }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6; border-radius: 8px 8px 0 0;
        padding: 8px 16px; font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e94560; color: white;
    }
    .section-ref {
        background: #eef2ff; border-left: 3px solid #6366f1;
        padding: 8px 12px; border-radius: 0 6px 6px 0;
        font-size: 0.85em; color: #4338ca; margin-bottom: 1rem;
    }
    .math-box {
        background: #fefce8; border: 1px solid #fbbf24;
        padding: 12px; border-radius: 8px; margin: 8px 0;
    }
    .insight-box {
        background: #f0fdf4; border: 1px solid #22c55e;
        padding: 12px; border-radius: 8px; margin: 8px 0;
    }
    div[data-testid="stMetricValue"] { font-size: 1.4rem; }
</style>
""", unsafe_allow_html=True)

# ─── Plotly Theme ──────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    font=dict(family="Inter, system-ui, sans-serif", size=12),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=40, r=20, t=40, b=40),
)
CLASS_COLORS = px.colors.qualitative.Set2

# ─── Core Math (from scratch) ──────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer."""
    return re.findall(r"[a-z]+", text.lower())

def build_vocab(all_samples: list[str]) -> list[str]:
    all_tokens = []
    for s in all_samples:
        all_tokens.extend(tokenize(s))
    return sorted(set(all_tokens))

def text_to_features(text: str, vocab: list[str]) -> np.ndarray:
    tokens = tokenize(text)
    counts = Counter(tokens)
    return np.array([counts.get(w, 0) for w in vocab], dtype=float)

def sigmoid(z):
    """Numerically stable sigmoid: σ(z) = 1 / (1 + exp(-z))"""
    z = np.clip(z, -500, 500)
    return np.where(z >= 0,
                    1 / (1 + np.exp(-z)),
                    np.exp(z) / (1 + np.exp(z)))

def softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable softmax: shift by max(z) before exp."""
    z_shifted = z - np.max(z)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z)

def cross_entropy_loss(probs: np.ndarray, true_idx: int) -> float:
    return -np.log(probs[true_idx] + 1e-12)

def forward(x: np.ndarray, W: np.ndarray, b: np.ndarray):
    z = W @ x + b
    return z, softmax(z)

def compute_total_loss(X_list, y_list, W, b, l2_lambda=0.0):
    total = 0.0
    for x, y in zip(X_list, y_list):
        _, probs = forward(x, W, b)
        total += cross_entropy_loss(probs, y)
    avg_loss = total / len(X_list)
    if l2_lambda > 0:
        avg_loss += l2_lambda * np.sum(W ** 2)
    return avg_loss

def sgd_step(x, true_idx, W, b, lr, num_classes, l2_lambda=0.0):
    """One SGD step. Returns new W, b, probs, loss, and the gradient info."""
    z, probs = forward(x, W, b)
    y_one_hot = np.zeros(num_classes)
    y_one_hot[true_idx] = 1.0
    error = probs - y_one_hot  # (ŷ - y) for each class

    grad_W = np.outer(error, x)
    grad_b = error.copy()

    # L2 regularization gradient
    if l2_lambda > 0:
        grad_W += 2 * l2_lambda * W

    W_new = W - lr * grad_W
    b_new = b - lr * grad_b
    loss = cross_entropy_loss(probs, true_idx)

    return W_new, b_new, probs, loss, error, grad_W, grad_b


# ─── Default Intents ───────────────────────────────────────────────────────────
DEFAULT_INTENTS = {
    "Check Balance": [
        "what is my account balance",
        "how much money do I have",
        "show me my balance",
        "what is my current balance",
        "can you check my balance",
    ],
    "Transfer Funds": [
        "I want to transfer money",
        "send money to another account",
        "transfer funds to savings",
        "move money between accounts",
        "I need to send a payment",
    ],
    "Fund Performance": [
        "how is my fund performing",
        "show me the returns on my investments",
        "what are the returns for vanguard index fund",
        "how did my portfolio do this year",
        "show fund performance data",
    ],
    "Account Help": [
        "I need help with my account",
        "how do I update my information",
        "I forgot my password",
        "change my email address",
        "how do I close my account",
    ],
    "Tax Documents": [
        "where are my tax forms",
        "I need my 1099 document",
        "download my tax statement",
        "when will my tax forms be ready",
        "I need tax documents for filing",
    ],
}


# ─── Session State ─────────────────────────────────────────────────────────────
def init_session():
    defaults = {
        "intents": {k: list(v) for k, v in DEFAULT_INTENTS.items()},
        "W": None, "b": None, "vocab": None,
        "loss_history": [], "epoch_count": 0,
        "step_log": [],  # detailed log of SGD steps
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

def build_dataset(intents, vocab):
    class_names = list(intents.keys())
    X_list, y_list = [], []
    for idx, (intent, samples) in enumerate(intents.items()):
        for s in samples:
            X_list.append(text_to_features(s, vocab))
            y_list.append(idx)
    return X_list, y_list, class_names

def reset_model(intents):
    all_samples = [s for samples in intents.values() for s in samples]
    vocab = build_vocab(all_samples)
    num_classes = len(intents)
    st.session_state.W = np.zeros((num_classes, len(vocab)))
    st.session_state.b = np.zeros(num_classes)
    st.session_state.vocab = vocab
    st.session_state.loss_history = []
    st.session_state.epoch_count = 0
    st.session_state.step_log = []
    return vocab


# ─── Header ────────────────────────────────────────────────────────────────────
st.title("📘 Logistic Regression Explorer")
st.caption(
    "Interactive companion to **Jurafsky & Martin, Ch. 4** (Sections 4.1–4.8) · "
    "All algorithms from scratch in NumPy"
)

# ─── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "① Sigmoid & Intuition",
    "② Setup Intents",
    "③ Feature Matrix",
    "④ Softmax Forward Pass",
    "⑤ Train (SGD)",
    "⑥ Predict & Evaluate",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: Sigmoid & Intuition
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.header("The Sigmoid Function & Logistic Regression Intuition")
    st.markdown('<div class="section-ref">📖 J&M Sections 4.1 – 4.3: Classification, the Sigmoid Function, Decision Boundaries</div>', unsafe_allow_html=True)

    st.markdown("""
    Before we jump to multinomial (multi-class) classification, let's build intuition
    with **binary logistic regression** — the foundation everything else rests on.

    The core idea: take a weighted sum of features **z = w·x + b**, then squeeze it
    through the **sigmoid function** σ(z) to get a probability between 0 and 1.
    """)

    st.latex(r"\sigma(z) = \frac{1}{1 + e^{-z}}")

    # Interactive sigmoid
    col_sig1, col_sig2 = st.columns([2, 1])
    with col_sig2:
        z_range = st.slider("z range", 1.0, 20.0, 10.0, step=0.5,
                            help="Horizontal extent of the plot")
        z_point = st.slider("Probe a point z =", -10.0, 10.0, 0.0, step=0.1)
        sig_val = sigmoid(np.array([z_point]))[0]
        st.metric("σ(z)", f"{sig_val:.4f}")
        st.metric("1 − σ(z)", f"{1 - sig_val:.4f}")

        st.markdown("---")
        st.markdown("""
        **Key properties:**
        - σ(0) = 0.5 (the decision boundary)
        - As z → +∞, σ(z) → 1
        - As z → −∞, σ(z) → 0
        - 1 − σ(z) = σ(−z)
        """)

    with col_sig1:
        z_vals = np.linspace(-z_range, z_range, 400)
        sig_vals = sigmoid(z_vals)

        fig_sig = go.Figure()
        fig_sig.add_trace(go.Scatter(
            x=z_vals, y=sig_vals, mode="lines",
            line=dict(color="#6366f1", width=3), name="σ(z)"
        ))
        # Decision boundary
        fig_sig.add_hline(y=0.5, line_dash="dash", line_color="#94a3b8",
                          annotation_text="Decision boundary (0.5)")
        # Probe point
        fig_sig.add_trace(go.Scatter(
            x=[z_point], y=[sig_val], mode="markers",
            marker=dict(color="#e94560", size=14, line=dict(width=2, color="white")),
            name=f"z={z_point:.1f} → σ={sig_val:.4f}"
        ))
        fig_sig.update_layout(
            **PLOTLY_LAYOUT,
            height=400,
            xaxis_title="z = w·x + b (logit score)",
            yaxis_title="σ(z) = P(y = 1 | x)",
            yaxis=dict(range=[-0.05, 1.05]),
            legend=dict(x=0.02, y=0.98),
        )
        st.plotly_chart(fig_sig, use_container_width=True)

    # Binary classification worked example
    st.markdown("---")
    st.subheader("Worked Example: Binary Sentiment Classification")
    st.markdown('<div class="section-ref">📖 J&M Section 4.3.1 — Sentiment classification with 6 features</div>', unsafe_allow_html=True)

    st.markdown("""
    The textbook uses a movie review with 6 hand-crafted features.
    Try adjusting the weights to see how they affect the classification:
    """)

    features_info = [
        ("x₁: Positive lexicon count", 3, 2.5),
        ("x₂: Negative lexicon count", 2, -5.0),
        ("x₃: Contains 'no'", 1, -1.2),
        ("x₄: Pronoun count", 3, 0.5),
        ("x₅: Contains '!'", 0, 2.0),
        ("x₆: ln(word count)", 4.19, 0.7),
    ]

    col_feat, col_wt, col_res = st.columns([1, 1, 1])

    with col_feat:
        st.markdown("**Feature values (from Fig. 4.2)**")
        x_vals = []
        for name, default, _ in features_info:
            v = st.number_input(name, value=float(default), step=0.1, format="%.2f",
                                key=f"bin_{name}")
            x_vals.append(v)

    with col_wt:
        st.markdown("**Learned weights**")
        w_vals = []
        for name, _, default_w in features_info:
            w = st.number_input(f"w for {name[:3]}", value=float(default_w),
                                step=0.1, format="%.2f", key=f"binw_{name}")
            w_vals.append(w)

    with col_res:
        bias_val = st.number_input("Bias b", value=0.1, step=0.01, format="%.2f")

        x_arr = np.array(x_vals)
        w_arr = np.array(w_vals)
        z_score = np.dot(w_arr, x_arr) + bias_val
        p_pos = sigmoid(np.array([z_score]))[0]

        st.markdown("---")
        st.markdown(f"**z = w·x + b = {z_score:.4f}**")
        st.markdown(f"**σ(z) = P(positive) = {p_pos:.4f}**")
        st.markdown(f"**P(negative) = {1 - p_pos:.4f}**")

        if p_pos >= 0.5:
            st.success(f"✅ Prediction: **Positive** ({p_pos*100:.1f}%)")
        else:
            st.error(f"❌ Prediction: **Negative** ({(1-p_pos)*100:.1f}%)")

        # Show dot product breakdown
        with st.expander("Show w·x term-by-term"):
            terms = []
            for i, (name, xv, _) in enumerate(features_info):
                product = w_vals[i] * x_vals[i]
                terms.append(product)
                sign = "+" if product >= 0 else ""
                st.markdown(f"`{w_vals[i]:+.2f}` × `{x_vals[i]:.2f}` = `{sign}{product:.2f}`  ← {name}")
            st.markdown(f"**Sum = {sum(terms):.4f}**, + b = {bias_val:.2f} → **z = {z_score:.4f}**")

    st.markdown("""
    <div class="insight-box">
    <strong>💡 Key Insight:</strong> Each weight captures how important a feature is for the positive class.
    Positive weights (like w₁ for positive words) push toward P(positive) → 1.
    Negative weights (like w₂ for negative words) push toward P(positive) → 0.
    The sigmoid squashes the unbounded score z into a valid probability.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ---
    **From Binary → Multinomial:** When we have more than 2 classes (for example intent-based classification), we replace the single weight vector **w** with a weight
    matrix **W** (one row per class), and replace sigmoid with **softmax**.
    The rest of the tabs explore this multinomial case.
    """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: Setup Intents
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.header("Define Your Intents and Training Samples")
    st.markdown('<div class="section-ref">📖 J&M Section 4.1 — Supervised classification with labeled training data</div>', unsafe_allow_html=True)

    st.markdown("""
    Each **intent** is a class (K classes total). Each sentence is a training sample
    with a known label. Together these form the labeled training set
    {(x⁽¹⁾, y⁽¹⁾), (x⁽²⁾, y⁽²⁾), ..., (x⁽ᵐ⁾, y⁽ᵐ⁾)}.
    """)

    intents_copy = {k: list(v) for k, v in st.session_state.intents.items()}

    col_add, col_remove = st.columns(2)
    with col_add:
        new_name = st.text_input("New intent name", placeholder="e.g. Open Account")
        if st.button("➕ Add Intent") and new_name.strip():
            name = new_name.strip()
            if name not in intents_copy:
                intents_copy[name] = ["sample sentence here"]
                st.session_state.intents = intents_copy
                st.rerun()
    with col_remove:
        if len(intents_copy) > 2:
            to_remove = st.selectbox("Select intent to remove", list(intents_copy.keys()))
            if st.button("🗑️ Remove Intent"):
                del intents_copy[to_remove]
                st.session_state.intents = intents_copy
                st.rerun()

    updated_intents = {}
    cols = st.columns(min(len(intents_copy), 3))
    for i, (intent_name, samples) in enumerate(intents_copy.items()):
        with cols[i % len(cols)]:
            st.markdown(f"**{intent_name}** ({len(samples)} samples)")
            raw = st.text_area(
                f"Samples for {intent_name}",
                value="\n".join(samples), height=160,
                key=f"intent_{intent_name}", label_visibility="collapsed",
            )
            lines = [l.strip() for l in raw.strip().split("\n") if l.strip()]
            updated_intents[intent_name] = lines

    st.session_state.intents = updated_intents

    st.divider()
    all_samples_flat = [s for v in updated_intents.values() for s in v]
    vocab_preview = build_vocab(all_samples_flat)

    col_stats, col_btn = st.columns([3, 1])
    with col_stats:
        st.markdown(
            f"**K = {len(updated_intents)} classes** · "
            f"**m = {len(all_samples_flat)} training samples** · "
            f"**|V| = {len(vocab_preview)} vocabulary tokens**"
        )
    with col_btn:
        if st.button("🔄 Initialize / Reset Model", type="primary"):
            reset_model(updated_intents)
            st.success("Model initialized with zero weights.")

    if st.session_state.W is not None:
        st.success(
            f"✅ Model ready — **W** shape: [{st.session_state.W.shape[0]} × "
            f"{st.session_state.W.shape[1]}] (K classes × |V| features)"
        )
    else:
        st.info("Click **Initialize / Reset Model** to build vocabulary and set up weight matrix.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: Feature Matrix
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.header("Feature Matrix — Bag of Words")
    st.markdown('<div class="section-ref">📖 J&M Section 4.1 — Feature representation: each input is a vector [x₁, x₂, ..., xₙ]</div>', unsafe_allow_html=True)

    st.markdown("""
    Each training sentence is converted into a **feature vector** by counting word
    occurrences. This is the bag-of-words representation — the input **x** to the classifier.
    """)

    if st.session_state.vocab is None:
        st.warning("⬅️ Initialize the model on the **Setup Intents** tab first.")
    else:
        vocab = st.session_state.vocab
        intents = st.session_state.intents
        X_list, y_list, class_names = build_dataset(intents, vocab)

        feature_matrix = np.array(X_list)
        labels = [class_names[y] for y in y_list]
        sample_texts = [s for samples in intents.values() for s in samples]

        st.markdown(f"**Vocabulary** ({len(vocab)} tokens): `{' · '.join(vocab[:50])}`"
                    + (f" ... (+{len(vocab)-50} more)" if len(vocab) > 50 else ""))

        st.markdown("#### Feature Heatmap")
        st.caption("Rows = training samples, Columns = vocabulary words. Cell = word count in that sample.")

        short_labels = [f"{labels[i][:10]}|{i}" for i in range(len(labels))]
        n_show = min(len(vocab), 60)
        vocab_display = vocab[:n_show]
        matrix_display = feature_matrix[:, :n_show]

        fig_fm = go.Figure(data=go.Heatmap(
            z=matrix_display, x=vocab_display, y=short_labels,
            colorscale="Blues", showscale=True,
            text=matrix_display.astype(int), texttemplate="%{text}",
        ))
        fig_fm.update_layout(
            **PLOTLY_LAYOUT,
            height=max(300, len(labels) * 22 + 100),
            xaxis_title="Vocabulary Token", yaxis_title="Training Sample",
            xaxis_tickangle=-45,
        )
        st.plotly_chart(fig_fm, use_container_width=True)

        with st.expander("🔍 Inspect a specific sample's feature vector"):
            sel = st.selectbox("Select sample", range(len(sample_texts)),
                               format_func=lambda i: f"[{labels[i]}] {sample_texts[i]}")
            fv = feature_matrix[sel]
            nonzero = [(vocab[j], int(fv[j])) for j in range(len(vocab)) if fv[j] > 0]
            st.markdown(f"**Sentence:** `{sample_texts[sel]}`")
            st.markdown(f"**Active features** (non-zero):")
            if nonzero:
                st.table({"Token": [x[0] for x in nonzero], "Count": [x[1] for x in nonzero]})
            st.markdown(f"**Full vector** (length {len(vocab)}): `{fv.astype(int).tolist()}`")

        # Show class-level vocabulary distribution
        with st.expander("📊 Which words are distinctive per class?"):
            st.caption("Sum of word counts across all samples in each class — helps see which words are 'signals' for each intent.")
            class_word_sums = {}
            for idx, cname in enumerate(class_names):
                mask = [i for i, y in enumerate(y_list) if y == idx]
                class_word_sums[cname] = feature_matrix[mask].sum(axis=0)

            for cname in class_names:
                sums = class_word_sums[cname]
                top_idx = np.argsort(sums)[::-1][:8]
                top_words = [(vocab[j], int(sums[j])) for j in top_idx if sums[j] > 0]
                st.markdown(f"**{cname}:** " + ", ".join(f"`{w}` ({c})" for w, c in top_words))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: Softmax Forward Pass
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.header("Softmax Forward Pass — Step by Step")
    st.markdown('<div class="section-ref">📖 J&M Sections 4.7 – 4.8: Softmax function, multinomial logistic regression, cross-entropy loss</div>', unsafe_allow_html=True)

    st.markdown("""
    For **K > 2** classes, we replace sigmoid with **softmax**. Pick any sample and
    trace the full computation: **z = Wx + b** → **softmax(z)** → **cross-entropy loss**.
    """)

    if st.session_state.W is None:
        st.warning("⬅️ Initialize the model on the **Setup Intents** tab first.")
    else:
        W = st.session_state.W
        b = st.session_state.b
        vocab = st.session_state.vocab
        intents = st.session_state.intents
        class_names = list(intents.keys())
        num_classes = len(class_names)

        input_mode = st.radio("Input mode", ["Pick a training sample", "Type your own sentence"],
                              horizontal=True)
        if input_mode == "Pick a training sample":
            all_samples = [s for samples in intents.values() for s in samples]
            all_labels = [c for c, samples in intents.items() for _ in samples]
            sel_idx = st.selectbox("Training sample", range(len(all_samples)),
                                   format_func=lambda i: f"[{all_labels[i]}] {all_samples[i]}")
            sentence = all_samples[sel_idx]
            true_label = all_labels[sel_idx]
        else:
            sentence = st.text_input("Type a sentence", value="what is my balance")
            true_label = None

        x = text_to_features(sentence, vocab)
        tokens_found = [vocab[i] for i in range(len(vocab)) if x[i] > 0]
        tokens_oov = [t for t in tokenize(sentence) if t not in vocab]

        st.markdown(f"**Tokens in vocabulary:** `{tokens_found}`")
        if tokens_oov:
            st.caption(f"Out-of-vocabulary (ignored): `{tokens_oov}`")

        # ── Step 1: Logits ──
        st.divider()
        st.markdown("### Step 1 — Compute Logits: z = Wx + b")
        st.latex(r"z_k = \mathbf{w}_k \cdot \mathbf{x} + b_k \quad \text{for each class } k = 1, \ldots, K")

        z = W @ x + b

        fig_z = go.Figure(go.Bar(
            x=class_names, y=z,
            marker_color=[CLASS_COLORS[i % len(CLASS_COLORS)] for i in range(num_classes)],
            text=[f"{v:.4f}" for v in z], textposition="outside",
        ))
        fig_z.update_layout(**PLOTLY_LAYOUT, height=300,
                            yaxis_title="Logit score z_k",
                            title="Logit Scores (z = Wx + b)")
        st.plotly_chart(fig_z, use_container_width=True)

        with st.expander("🔍 Show w_k · x breakdown per class"):
            for c_idx, c_name in enumerate(class_names):
                dot = W[c_idx] @ x
                # Show which features contributed
                contributions = [(vocab[j], W[c_idx, j], x[j], W[c_idx, j] * x[j])
                                 for j in range(len(vocab)) if x[j] > 0 and W[c_idx, j] != 0]
                st.markdown(f"**{c_name}**: w·x = `{dot:.4f}` + b = `{b[c_idx]:.4f}` → z = `{z[c_idx]:.4f}`")
                if contributions:
                    for tok, wij, xj, prod in sorted(contributions, key=lambda t: -abs(t[3])):
                        st.caption(f"    {tok}: w={wij:.4f} × x={xj:.0f} = {prod:.4f}")
                st.markdown("---")

        # ── Step 2: Softmax ──
        st.divider()
        st.markdown("### Step 2 — Apply Softmax")
        st.latex(r"\text{softmax}(z_k) = \frac{\exp(z_k)}{\sum_{j=1}^{K} \exp(z_j)}")

        z_shifted = z - np.max(z)
        exp_z = np.exp(z_shifted)
        probs = exp_z / np.sum(exp_z)

        col_sm1, col_sm2 = st.columns([3, 2])
        with col_sm1:
            fig_sm = go.Figure(go.Bar(
                x=class_names, y=probs,
                marker_color=[CLASS_COLORS[i % len(CLASS_COLORS)] for i in range(num_classes)],
                text=[f"{v*100:.1f}%" for v in probs], textposition="outside",
            ))
            if true_label:
                fig_sm.add_hline(y=1.0/num_classes, line_dash="dash", line_color="#94a3b8",
                                 annotation_text=f"Random chance (1/{num_classes})")
            fig_sm.update_layout(**PLOTLY_LAYOUT, height=320,
                                 yaxis_title="P(class | x)", yaxis_range=[0, 1.15],
                                 title="Softmax Probabilities")
            st.plotly_chart(fig_sm, use_container_width=True)

        with col_sm2:
            st.markdown("#### Computation Detail")
            st.markdown(f"**1. Shift for stability:** z' = z − max(z)")
            st.code(f"z_shifted = {np.round(z_shifted, 4).tolist()}", language=None)
            st.markdown(f"**2. Exponentiate:** exp(z')")
            st.code(f"exp(z') = {np.round(exp_z, 6).tolist()}", language=None)
            st.markdown(f"**3. Sum:** Σ exp(z') = {np.sum(exp_z):.6f}")
            st.markdown(f"**4. Normalize:** softmax = exp(z') / Σ")
            st.code(f"P = {np.round(probs, 4).tolist()}", language=None)
            st.markdown(f"**5. Verify:** Σ P = {np.sum(probs):.6f} ✓")

        # ── Step 3: Cross-Entropy Loss ──
        if true_label:
            st.divider()
            st.markdown("### Step 3 — Cross-Entropy Loss")
            st.latex(r"L_{CE}(\hat{y}, y) = -\log \hat{y}_c = -\log P(y_c = 1 \mid x)")

            true_idx = class_names.index(true_label)
            true_prob = probs[true_idx]
            loss = -np.log(true_prob + 1e-12)
            pred_class = class_names[np.argmax(probs)]

            col_r1, col_r2, col_r3 = st.columns(3)
            col_r1.metric("Predicted Class", pred_class, f"{np.max(probs)*100:.1f}%")
            col_r2.metric("True Class", true_label, f"P = {true_prob:.4f}")
            col_r3.metric("Cross-Entropy Loss", f"{loss:.4f}",
                          "✓ correct" if pred_class == true_label else "✗ incorrect")

            # Loss intuition
            st.markdown("""
            <div class="math-box">
            <strong>📐 Intuition:</strong> Cross-entropy loss = −log(probability assigned to the correct class).
            If the model assigns P = 1.0 to the right class → loss = 0 (perfect).
            If the model assigns P → 0 to the right class → loss → ∞ (terrible).
            This is the signal that drives gradient descent to improve the weights.
            </div>
            """, unsafe_allow_html=True)

            # Loss curve visualization
            with st.expander("📈 Visualize the −log(p) loss curve"):
                p_vals = np.linspace(0.001, 1.0, 200)
                loss_vals = -np.log(p_vals)
                fig_loss_curve = go.Figure()
                fig_loss_curve.add_trace(go.Scatter(
                    x=p_vals, y=loss_vals, mode="lines",
                    line=dict(color="#e94560", width=3), name="−log(p)"
                ))
                fig_loss_curve.add_trace(go.Scatter(
                    x=[true_prob], y=[loss], mode="markers",
                    marker=dict(color="#6366f1", size=14, line=dict(width=2, color="white")),
                    name=f"Current: P={true_prob:.3f}, L={loss:.3f}"
                ))
                fig_loss_curve.update_layout(
                    **PLOTLY_LAYOUT, height=280,
                    xaxis_title="P(correct class | x)",
                    yaxis_title="Cross-Entropy Loss",
                    title="Loss as a function of P(correct class)"
                )
                st.plotly_chart(fig_loss_curve, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: Training — SGD
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.header("Training — Stochastic Gradient Descent")
    st.markdown('<div class="section-ref">📖 J&M Sections 4.6 – 4.8: Gradient descent, SGD update rule, learning in multinomial LR</div>', unsafe_allow_html=True)

    st.markdown("""
    Training adjusts **W** and **b** to minimize cross-entropy loss. The gradient
    tells us *which direction* to move each weight, and the learning rate η
    controls *how far* to move.
    """)
    st.latex(r"W_k \leftarrow W_k - \eta \cdot (\hat{p}_k - y_k) \cdot \mathbf{x} \qquad b_k \leftarrow b_k - \eta \cdot (\hat{p}_k - y_k)")

    st.markdown("""
    <div class="math-box">
    <strong>📐 The gradient is beautifully simple:</strong> For each class k, the error is
    (ŷₖ − yₖ) — the difference between what the model predicted and the truth.
    Multiply by the input features x, and you get the weight update. The model
    nudges weights <em>toward</em> the true class and <em>away</em> from incorrect classes.
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.W is None:
        st.warning("⬅️ Initialize the model on the **Setup Intents** tab first.")
    else:
        vocab = st.session_state.vocab
        intents = st.session_state.intents
        X_list, y_list, class_names = build_dataset(intents, vocab)
        num_classes = len(class_names)

        # Controls
        col_c1, col_c2, col_c3, col_c4 = st.columns(4)
        with col_c1:
            lr = st.slider("Learning rate η", 0.001, 1.0, 0.1, step=0.001, format="%.3f")
        with col_c2:
            n_epochs = st.slider("Epochs to run", 1, 100, 10)
        with col_c3:
            shuffle = st.checkbox("Shuffle each epoch", value=True)
        with col_c4:
            l2_lambda = st.slider("L2 regularization λ", 0.0, 0.5, 0.0, step=0.01,
                                  help="J&M §4.10: Prevents overfitting by penalizing large weights")

        col_btn1, col_btn2, col_btn3 = st.columns(3)
        run_clicked = col_btn1.button(f"▶️ Run {n_epochs} Epoch(s)", type="primary")
        step_clicked = col_btn2.button("🔬 Single SGD Step (with anatomy)")
        reset_clicked = col_btn3.button("🔄 Reset Weights to Zero")

        if reset_clicked:
            reset_model(intents)
            st.rerun()

        W = st.session_state.W
        b = st.session_state.b
        loss_history = st.session_state.loss_history
        epoch_count = st.session_state.epoch_count

        # ── Single SGD Step with full anatomy ──
        if step_clicked:
            rand_idx = np.random.randint(len(X_list))
            x_step = X_list[rand_idx]
            y_step = y_list[rand_idx]

            W_old = W.copy()
            b_old = b.copy()

            W, b, probs_step, step_loss, error, grad_W, grad_b = sgd_step(
                x_step, y_step, W, b, lr, num_classes, l2_lambda)

            loss_history.append(step_loss)
            st.session_state.W = W
            st.session_state.b = b
            st.session_state.loss_history = loss_history

            sample_text = [s for samples in intents.values() for s in samples][rand_idx]

            st.info(f"**SGD step on:** \"{sample_text}\" (true: **{class_names[y_step]}**) — loss: **{step_loss:.4f}**")

            # Gradient Anatomy
            st.markdown("#### 🔬 Gradient Anatomy for This Step")

            # Show error vector
            y_one_hot = np.zeros(num_classes)
            y_one_hot[y_step] = 1.0

            col_g1, col_g2 = st.columns(2)
            with col_g1:
                st.markdown("**Error vector (ŷ − y):**")
                for k in range(num_classes):
                    err_k = probs_step[k] - y_one_hot[k]
                    marker = " ← true class" if k == y_step else ""
                    color = "🟢" if abs(err_k) < 0.1 else "🔴"
                    st.markdown(f"{color} **{class_names[k]}**: ŷ={probs_step[k]:.4f} − y={y_one_hot[k]:.0f} = **{err_k:+.4f}**{marker}")

            with col_g2:
                st.markdown("**What this means:**")
                st.markdown(f"""
                - For the **true class** ({class_names[y_step]}), error = ŷ − 1 = **{probs_step[y_step] - 1:+.4f}** (negative → weights move *toward* this class)
                - For **other classes**, error = ŷ − 0 = positive → weights move *away*
                - Multiplied by active features x and learning rate η = {lr}
                """)

            # Show weight changes
            with st.expander("📊 Weight changes ΔW (top movers)"):
                delta_W = W - W_old
                delta_b = b - b_old

                for k in range(num_classes):
                    changes = [(vocab[j], delta_W[k, j], W_old[k, j], W[k, j])
                               for j in range(len(vocab)) if abs(delta_W[k, j]) > 1e-8]
                    changes.sort(key=lambda t: -abs(t[1]))
                    if changes:
                        st.markdown(f"**{class_names[k]}** (Δb = {delta_b[k]:+.6f}):")
                        for tok, dw, old_w, new_w in changes[:5]:
                            direction = "↑" if dw > 0 else "↓"
                            st.caption(f"    {direction} `{tok}`: {old_w:.4f} → {new_w:.4f} (Δ = {dw:+.6f})")

        # ── Run N Epochs ──
        if run_clicked:
            indices = list(range(len(X_list)))
            for epoch in range(n_epochs):
                if shuffle:
                    np.random.shuffle(indices)
                epoch_loss = 0.0
                for i in indices:
                    W, b, _, step_loss, _, _, _ = sgd_step(
                        X_list[i], y_list[i], W, b, lr, num_classes, l2_lambda)
                    epoch_loss += step_loss
                loss_history.append(epoch_loss / len(X_list))
                epoch_count += 1
            st.session_state.W = W
            st.session_state.b = b
            st.session_state.loss_history = loss_history
            st.session_state.epoch_count = epoch_count

        # ── Metrics ──
        st.divider()
        total_loss = compute_total_loss(X_list, y_list, W, b, l2_lambda)
        preds = [class_names[np.argmax(forward(x, W, b)[1])] for x in X_list]
        accuracy = np.mean([preds[i] == class_names[y_list[i]] for i in range(len(y_list))])

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Avg Cross-Entropy Loss", f"{total_loss:.4f}")
        mc2.metric("Training Accuracy", f"{accuracy*100:.1f}%")
        mc3.metric("Epochs Completed", epoch_count)

        # ── Loss Curve ──
        if loss_history:
            st.markdown("#### Loss Curve")
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                y=loss_history, mode="lines+markers",
                line=dict(color="#e94560", width=2),
                marker=dict(size=3), name="Loss"
            ))
            fig_loss.update_layout(**PLOTLY_LAYOUT, height=280,
                                   xaxis_title="Step / Epoch",
                                   yaxis_title="Cross-Entropy Loss")
            st.plotly_chart(fig_loss, use_container_width=True)

        # ── Weight Matrix Heatmap ──
        st.markdown("#### Weight Matrix W")
        st.caption("Each row = a class prototype. Positive weights (blue) → evidence FOR that class. Negative (red) → evidence AGAINST.")

        n_show = min(len(vocab), 60)
        vocab_display = vocab[:n_show]
        W_display = W[:, :n_show]
        abs_max = max(np.max(np.abs(W_display)), 0.001)

        fig_w = go.Figure(data=go.Heatmap(
            z=W_display, x=vocab_display, y=class_names,
            colorscale="RdBu", zmid=0, zmin=-abs_max, zmax=abs_max,
            text=np.round(W_display, 3), texttemplate="%{text}",
            showscale=True,
        ))
        fig_w.update_layout(**PLOTLY_LAYOUT,
                            height=max(250, num_classes * 60 + 100),
                            xaxis_tickangle=-45)
        st.plotly_chart(fig_w, use_container_width=True)

        # ── Bias Vector ──
        col_bias, col_topw = st.columns([1, 2])
        with col_bias:
            st.markdown("#### Bias Vector b")
            fig_b = go.Figure(go.Bar(
                x=class_names, y=b,
                marker_color=["#4C78A8" if v >= 0 else "#E45756" for v in b],
                text=[f"{v:.4f}" for v in b], textposition="outside",
            ))
            fig_b.update_layout(**PLOTLY_LAYOUT, height=260, yaxis_title="Bias value")
            st.plotly_chart(fig_b, use_container_width=True)

        with col_topw:
            st.markdown("#### Top Discriminative Features per Class")
            for c_idx, c_name in enumerate(class_names):
                w_row = W[c_idx]
                top_pos = sorted(zip(vocab, w_row), key=lambda x: -x[1])[:3]
                top_neg = sorted(zip(vocab, w_row), key=lambda x: x[1])[:3]
                pos_str = ", ".join(f"`{t}` ({w:.3f})" for t, w in top_pos if w != 0)
                neg_str = ", ".join(f"`{t}` ({w:.3f})" for t, w in top_neg if w != 0)
                st.markdown(f"**{c_name}** — 🟢 FOR: {pos_str or 'none yet'} · 🔴 AGAINST: {neg_str or 'none yet'}")

        # Regularization note
        if l2_lambda > 0:
            st.markdown(f"""
            <div class="insight-box">
            <strong>🛡️ L2 Regularization active (λ = {l2_lambda})</strong><br>
            The loss now includes a penalty term λ·‖W‖² that discourages large weights.
            This prevents overfitting by keeping the model from relying too heavily on any
            single feature. (J&M Section 4.10)
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6: Predict & Evaluate
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.header("Predict & Evaluate")
    st.markdown('<div class="section-ref">📖 J&M Sections 4.7.2, 4.9: Applying softmax for prediction; Precision, Recall, F-measure</div>', unsafe_allow_html=True)

    if st.session_state.W is None:
        st.warning("⬅️ Initialize and train the model first.")
    else:
        W = st.session_state.W
        b = st.session_state.b
        vocab = st.session_state.vocab
        intents = st.session_state.intents
        class_names = list(intents.keys())
        num_classes = len(class_names)

        # ── Prediction ──
        st.subheader("Classify a New Sentence")
        test_sentence = st.text_input(
            "Enter a sentence to classify",
            value="can I see my account balance please",
            placeholder="Type an intent-related sentence...",
        )

        if test_sentence.strip():
            x = text_to_features(test_sentence, vocab)
            z, probs = forward(x, W, b)
            pred_class = class_names[np.argmax(probs)]
            pred_prob = np.max(probs)

            tokens_found = [vocab[i] for i in range(len(vocab)) if x[i] > 0]
            tokens_missing = [t for t in tokenize(test_sentence) if t not in vocab]

            col_r, col_l = st.columns([1, 2])
            with col_r:
                st.metric("Predicted Intent", pred_class)
                st.metric("Confidence", f"{pred_prob*100:.1f}%")
                if tokens_found:
                    st.markdown(f"**Matched tokens:** `{tokens_found}`")
                if tokens_missing:
                    st.caption(f"OOV (ignored): `{tokens_missing}`")

            with col_l:
                sorted_idx = np.argsort(probs)[::-1]
                fig_pred = go.Figure(go.Bar(
                    x=[class_names[i] for i in sorted_idx],
                    y=[probs[i] for i in sorted_idx],
                    marker_color=[CLASS_COLORS[i % len(CLASS_COLORS)] for i in sorted_idx],
                    text=[f"{probs[i]*100:.1f}%" for i in sorted_idx],
                    textposition="outside",
                ))
                fig_pred.update_layout(**PLOTLY_LAYOUT, height=300,
                                       yaxis_title="P(class | x)", yaxis_range=[0, 1.15],
                                       title="Probability Breakdown")
                st.plotly_chart(fig_pred, use_container_width=True)

            with st.expander("🔍 Full softmax computation"):
                exp_z_pred = np.exp(z - np.max(z))
                st.code(f"z (logits)     = {np.round(z, 4).tolist()}\n"
                        f"z - max(z)     = {np.round(z - np.max(z), 4).tolist()}\n"
                        f"exp(z_shifted) = {np.round(exp_z_pred, 6).tolist()}\n"
                        f"sum(exp)       = {np.sum(exp_z_pred):.6f}\n"
                        f"softmax        = {np.round(probs, 4).tolist()}\n"
                        f"sum(softmax)   = {np.sum(probs):.6f}", language=None)

            # Logit scores
            st.markdown("#### Logit Scores (z = Wx + b)")
            fig_logits = go.Figure(go.Bar(
                x=class_names, y=z,
                marker_color=["#4C78A8" if v >= 0 else "#E45756" for v in z],
                text=[f"{v:.4f}" for v in z], textposition="outside",
            ))
            fig_logits.update_layout(**PLOTLY_LAYOUT, height=260, yaxis_title="Logit score z_k")
            st.plotly_chart(fig_logits, use_container_width=True)

        # ── Confusion Matrix ──
        st.divider()
        st.subheader("Training Set Evaluation — Confusion Matrix")
        st.caption("How well does the current model classify the training data? This ties into J&M Section 4.9 (Precision, Recall, F-measure).")

        X_list, y_list, _ = build_dataset(intents, vocab)
        y_pred = [np.argmax(forward(x, W, b)[1]) for x in X_list]

        # Build confusion matrix
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for true, pred in zip(y_list, y_pred):
            cm[true][pred] += 1

        fig_cm = go.Figure(data=go.Heatmap(
            z=cm, x=class_names, y=class_names,
            colorscale="Blues", showscale=True,
            text=cm, texttemplate="%{text}",
        ))
        fig_cm.update_layout(
            **PLOTLY_LAYOUT, height=max(300, num_classes * 60 + 80),
            xaxis_title="Predicted", yaxis_title="True",
            title="Confusion Matrix (Training Set)"
        )
        st.plotly_chart(fig_cm, use_container_width=True)

        # Per-class metrics
        st.markdown("#### Per-Class Precision, Recall, F1")
        metrics_data = []
        for k in range(num_classes):
            tp = cm[k, k]
            fp = cm[:, k].sum() - tp
            fn = cm[k, :].sum() - tp
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            metrics_data.append({
                "Intent": class_names[k],
                "TP": tp, "FP": fp, "FN": fn,
                "Precision": f"{precision:.3f}",
                "Recall": f"{recall:.3f}",
                "F1": f"{f1:.3f}",
            })

        st.dataframe(metrics_data, use_container_width=True, hide_index=True)

        # Macro vs Micro F1
        all_tp = sum(cm[k, k] for k in range(num_classes))
        micro_p = all_tp / max(sum(cm[:, k].sum() for k in range(num_classes)), 1)
        micro_r = all_tp / max(sum(cm[k, :].sum() for k in range(num_classes)), 1)
        micro_f1 = 2 * micro_p * micro_r / max(micro_p + micro_r, 1e-12)

        f1_vals = []
        for k in range(num_classes):
            tp = cm[k, k]; fp = cm[:, k].sum() - tp; fn = cm[k, :].sum() - tp
            p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
            f1_vals.append(2 * p * r / max(p + r, 1e-12))
        macro_f1 = np.mean(f1_vals)

        col_mf1, col_mif1 = st.columns(2)
        col_mf1.metric("Macro F1", f"{macro_f1:.3f}",
                        help="Average F1 across classes — treats all classes equally")
        col_mif1.metric("Micro F1", f"{micro_f1:.3f}",
                         help="F1 computed from total TP/FP/FN — weighted by class frequency")

        if abs(macro_f1 - micro_f1) > 0.1:
            st.markdown("""
            <div class="insight-box">
            <strong>💡 Macro vs. Micro F1 gap detected!</strong> This is the same phenomenon
            you've seen in Echo — when classes are imbalanced or some intents are harder
            to learn, macro F1 drops while micro F1 stays high. Macro F1 treats every class
            equally, exposing weaknesses on rare or confusable intents.
            </div>
            """, unsafe_allow_html=True)


# ─── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Built for learning · Based on Jurafsky & Martin, *Speech and Language Processing*, "
    "Chapter 4 (Sections 4.1–4.8) · All algorithms implemented from scratch in NumPy"
)
