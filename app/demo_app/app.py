import torch
torch.classes = None

import os, sys, pandas as pd, streamlit as st
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
pd.options.display.float_format = "{:.10f}".format 
import plotly.express as px

from utils.preprocessing          import preprocess_folktale
from utils.feature_engineering    import add_features_for_classification
from utils.prepare_sentence_level import build_sentence_level_dataset   # <-- keep if you still need it elsewhere

st.set_page_config(page_title="Klasifikasi Tokoh Cerita Rakyat", layout="centered")
st.title("🧙‍♀️ Klasifikasi Tokoh dalam Cerita Rakyat Nusantara")

st.caption("Dibuat oleh Rayssa Ravelia – NRP 5025211219")

st.subheader("Unggah atau tempelkan teks cerita rakyat untuk memulai")

# =========  INPUT AREA  =======================================================
title_input = st.text_input("📝 Judul Cerita  (Optional)", placeholder="contoh: Ikan Ajaib")

tab1, tab2 = st.tabs(["📁 Unggah File", "✍️ Tempelkan Teks"])
story_text = None
with tab1:
    up = st.file_uploader("Unggah file .txt", type=["txt"])
    if up: story_text = up.read().decode("utf-8")
with tab2:
    text = st.text_area("Tempelkan teks cerita rakyat di sini:", height=300)
    if text.strip(): story_text = text

# Sanitize title for filenames
clean_title = title_input.strip().lower().replace(" ", "_") or "uploaded"

# ============================================================================
if story_text:
    st.markdown("### 📖 Pratinjau Cerita Rakyat")
    st.write(story_text[:1000] + ("..." if len(story_text) > 1000 else ""))

    if st.button("➡️ Pra-pemrosesan & Klasifikasi"):
        with st.spinner("Sedang memproses cerita..."):
            title            = title_input.strip() or "uploaded"
            preprocessed_df  = preprocess_folktale(story_text, story_id=1, title=title)
            st.session_state["preprocessed_df"] = preprocessed_df
            st.success("Pra-pemrosesan selesai!")

    # -------------------------------------------------------------------------
    if "preprocessed_df" in st.session_state:
        preprocessed_df = st.session_state["preprocessed_df"]

        st.markdown("### 🧾 Pratinjau Tokenisasi (50 Baris Pertama)")
        st.dataframe(preprocessed_df.head(50))
        st.download_button("📥  Unduh CSV Tokenisasi",
                           preprocessed_df.to_csv(index=False).encode(),
                           f"tokenisasi_cerita_{clean_title}.csv", "text/csv")

        # ---------------------  NER Character Extraction  ---------------------
        from utils.predict import extract_characters
        sentences     = preprocessed_df.groupby("sentence_id")["word"].apply(list).tolist()
        character_df  = extract_characters(sentences)
        st.session_state["character_df"] = character_df

        st.markdown("### 🧑‍🤝‍🧑 Tokoh yang Terdeteksi (dengan Alias yang Dinormalisasi)")
        st.dataframe(character_df)

        if not character_df.empty:
            st.download_button("📥 Unduh Daftar Tokoh (CSV)",
                               character_df.to_csv(index=False).encode(),
                               f"daftar_tokoh_dinormalisasi_{clean_title}.csv", "text/csv")

            # --------------------  Alias Clustering  --------------------------
            from utils.alias_clustering import cluster_character_aliases
            clusters   = cluster_character_aliases(character_df["Normalized"].tolist())
            cluster_df = pd.DataFrame([{"Cluster": k, "Aliases": ", ".join(v)}
                                       for k, v in clusters.items()])
            st.markdown("### 🧩 Klaster Alias (Tokoh yang Dikelompokkan)")
            st.dataframe(cluster_df)
            st.download_button("📥 Unduh Klaster Alias (CSV)",
                               cluster_df.to_csv(index=False).encode(),
                               f"klaster_alias_{clean_title}.csv", "text/csv")

            # --------------------  Sense Mapping  -----------------------------
            from utils.sense_mapper import apply_role_based_merging
            alias_cluster_input = pd.DataFrame(
                {"story_id": 1, "person": k, "aliases": v} for k, v in clusters.items()
            )
            merged_df = apply_role_based_merging(alias_cluster_input)
            st.session_state["merged_df"] = merged_df

            # -----------  attach sentence_ids to every Tokoh  -----------------
            alias_sent_map = (
                character_df[["Normalized", "sentence_id"]]
                    .explode("Normalized").dropna()
                    .groupby("Normalized")["sentence_id"]
                    .agg(lambda x: sorted(set(x))).to_dict()
            )
            merged_df["sentence_ids"] = merged_df["aliases"].apply(
                lambda alist: sorted({sid
                                      for alias in alist
                                      for sid in alias_sent_map.get(alias, [])})
            )

            st.markdown("### 🧠 Klaster Berbasis Peran (Penggabungan Berdasarkan Peran Tokoh)")
            st.dataframe(merged_df)
            st.download_button("📥 Unduh Klaster Peran (CSV)",
                               merged_df.to_csv(index=False).encode(),
                               f"klaster_alias_berdasarkan_peran_{clean_title}.csv", "text/csv")

            # ------------------  Feature Engineering  ------------------------
            enriched_df = add_features_for_classification(merged_df, preprocessed_df)
            st.session_state["enriched_df"] = enriched_df

            st.markdown("### 🧬 Dataset Kalimat yang Telah Diperkaya (50 Baris Pertama)")
            st.dataframe(enriched_df.head(50))
            st.download_button("📥 Unduh Dataset Diperkaya (CSV)",
                               enriched_df.to_csv(index=False).encode(),
                               f"dataset_kalimat_diperkaya_{clean_title}.csv", "text/csv")

    # -------------------------------------------------------------------------
    if "enriched_df" in st.session_state:
        enriched_df = st.session_state["enriched_df"]
        merged_df   = st.session_state["merged_df"]

        st.markdown("### 🎭  Klasifikasi Jenis Tokoh")
        st.info("Pilih metode klasifikasi: **Machine Learning Klasik** (F1≈96 %) atau **Deep Learning (BERT)** (F1≈90 %).")

        # Define mapping between display label and internal tag
        model_display_to_tag = {
            "Machine Learning Klasik (Akurasi Terbaik)": "classical",
            "Deep Learning (BERT)": "bert"
        }
        model_display_options = list(model_display_to_tag.keys())
        tag_to_display = {v: k for k, v in model_display_to_tag.items()}

        # Set default tag if not already stored
        if "model_tag" not in st.session_state:
            st.session_state["model_tag"] = "classical"

        # Reverse map to find current display
        current_display = tag_to_display.get(st.session_state["model_tag"], model_display_options[0])

        # Selectbox and update tag
        selected_display = st.selectbox("Select a model:", model_display_options, index=model_display_options.index(current_display))
        st.session_state["model_tag"] = model_display_to_tag[selected_display]

        if st.button("🔍 Jalankan Klasifikasi Tokoh"):
            with st.spinner("Sedang mengklasifikasikan tokoh..."):
                if st.session_state["model_tag"] == "classical":
                    from utils.classical_classifier import classify_characters
                    prediction_df = classify_characters(enriched_df)

                    from utils.majority_vote import run_majority_vote
                    final_df = run_majority_vote(prediction_df, enriched_df)
                else:
                    from utils.bert_classifier import classify_characters
                    prediction_df = classify_characters(enriched_df)

                    from utils.confidence_vote import confidence_weighted_vote
                    final_df = confidence_weighted_vote(prediction_df)

            st.success("Klasifikasi selesai!")

            # Save to session state
            st.session_state["prediction_df"] = prediction_df
            st.session_state["final_df"] = final_df

        if "prediction_df" in st.session_state and "final_df" in st.session_state:
            prediction_df = st.session_state["prediction_df"]
            final_df = st.session_state["final_df"]
            model_tag = st.session_state["model_tag"]

            st.markdown("### 📌 Hasil Klasifikasi")
            st.dataframe(prediction_df)
            st.download_button("📥 Unduh Hasil Klasifikasi",
                            prediction_df.to_csv(index=False).encode(),
                            file_name=f"hasil_klasifikasi_tokoh_{clean_title}_{model_tag}.csv",
                            mime="text/csv")

            st.markdown("### 🏷️ Label Tokoh Final (Per Tokoh)")
            st.dataframe(final_df)
            st.download_button("📥 Unduh Label Tokoh Final",
                            final_df.to_csv(index=False).encode(),
                            file_name=f"label_tokoh_final_{clean_title}_{model_tag}.csv",
                            mime="text/csv")
            
            # ---------------------  Bar Chart  ---------------------

            # 1. Pastikan 'aliases' di final_df sebagai list Python
            # Jika ternyata berbentuk string "a, b, c", maka kita split menjadi list ["a", "b", "c"].
            def normalize_aliases(x):
                if isinstance(x, list):
                    # misalnya: ['ibu', 'mama']
                    return [str(a).strip() for a in x if str(a).strip()]
                elif isinstance(x, str):
                    # misalnya: "ibu, mama, ratu" -> ["ibu", "mama", "ratu"]
                    return [s.strip() for s in x.split(",") if s.strip()]
                else:
                    return []

            final_df["aliases_list"] = final_df["aliases"].apply(normalize_aliases)

            # 2. Ambil alias pertama (alias_utama) agar tooltip tidak terlalu panjang
            final_df["alias_utama"] = final_df["aliases_list"].apply(lambda lst: lst[0] if len(lst) > 0 else "-")

            # 3. Kita ingin menghitung jumlah tokoh per tipe, 
            #    sekaligus menggabungkan daftar alias_utama (atau bisa daftar lengkap) per tipe
            #    Kemudian akhirnya kita display di tooltip.
            chart_df = (
                final_df
                .groupby("predicted_type")
                .agg(
                    jumlah=("person", "count"),
                    daftar_alias=("alias_utama", lambda s: "<br>".join(sorted(set([a for a in s if a and a != "-"]))))
                )
                .reset_index()
            )

            # 4. Buat Plotly Bar Chart dengan tooltip yang menampilkan 'daftar_alias' secara baris ke bawah
            fig = px.bar(
                chart_df,
                x="predicted_type",
                y="jumlah",
                color="predicted_type",
                custom_data=["daftar_alias"],  # ini nanti akan dipanggil di hovertemplate
                labels={
                    "predicted_type": "Tipe Tokoh",
                    "jumlah": "Jumlah Tokoh"
                },
                title="📊 Statistik Tipe Tokoh"
            )

            # 5. Atur hovertemplate agar menampilkan daftar_alias (newline) dan jumlah
            fig.update_traces(
                hovertemplate=
                    "<b>%{x}</b><br>" +
                    "Alias:<br>%{customdata[0]}<br>" +
                    "Jumlah: %{y}<extra></extra>"
            )
            fig.update_layout(
                showlegend=False,
                xaxis_title="Tipe Tokoh",
                yaxis_title="Jumlah Tokoh"
            )

            st.markdown("### 📊 Statistik Tipe Tokoh")
            st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("Silakan unggah atau tempel cerita rakyat terlebih dahulu untuk melanjutkan.")