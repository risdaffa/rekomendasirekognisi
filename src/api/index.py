
import os
import re
from typing import Dict, List
from xml.dom import ValidationErr

import numpy as np
import pandas as pd

from .recommendation import extract_direct_recommendations

from .modeling import evaluate_LDA, preprocess_data_to_bow

from .preprocessing import apply_stemming, build_term_dict, casefolding_replace, clean_text, filter_and_translate, filter_stopwords, lemmatize_text, stemmed_wrapper, tokenize, ubah
from .similarity import calculate_cosine, calculate_directed_AB, calculate_directed_BA
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import logging

# Create FastAPI instance with custom docs and openapi url
app = FastAPI(docs_url="/api/py/docs", openapi_url="/api/py/openapi.json")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Helper function to read data from CSV
def read_data_mk():
    try:
        return pd.read_csv('./src/api/data/dataMK.csv')
    except Exception as e:
        logger.error(f"Error reading dataMK.csv: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error reading dataMK.csv: {str(e)}"
        )

# Helper function to write data to CSV


def write_data_mk(data_mk):
    try:
        data_mk.to_csv('./src/api/data/dataMK.csv', index=False)
    except Exception as e:
        logger.error(f"Error writing to dataMK.csv: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error writing to dataMK.csv: {str(e)}"
        )


@app.get("/api/py/matakuliah")
async def get_matakuliah():
    try:
        data_MK = pd.read_csv('./src/api/data/dataMK.csv')
        matakuliah_list = []
        for index, row in data_MK.iterrows():
            matakuliah_list.append({
                "nama": row['namaMK'],
                "deskripsi": row['deskripsiMK']
            })
        return {"matakuliah": matakuliah_list}
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )


@app.post("/api/py/matakuliah")
async def add_matakuliah(request: Request):
    try:
        body = await request.json()
        logger.debug(f"Adding course: {body}")
        data_MK = read_data_mk()
        new_course = {
            'namaMK': body['nama'],
            'deskripsiMK': body['deskripsi']
        }
        # Use concat to add the new course to the DataFrame
        data_MK = pd.concat(
            [data_MK, pd.DataFrame([new_course])], ignore_index=True)
        write_data_mk(data_MK)

        # Preprocess the new course data
        preprocessed_data = await get_preprocessed_data(request)
        if "error" in preprocessed_data:
            raise HTTPException(
                status_code=500,
                detail=f"Error preprocessing data: {preprocessed_data['error']}"
            )
        preprocessed_data_formated = {
            'namaMK': preprocessed_data['nama'],
            'deskripsiMK': preprocessed_data['deskripsi']
        }

        # Append the preprocessed data to preprocessed_dataMK.csv
        preprocessed_file_path = './src/api/data/preprocessed_dataMK.csv'
        if os.path.exists(preprocessed_file_path):
            existing_preprocessed_data = pd.read_csv(preprocessed_file_path)
            combined_preprocessed_data = pd.concat(
                [existing_preprocessed_data, pd.DataFrame(preprocessed_data_formated)], ignore_index=True)
            combined_preprocessed_data.to_csv(
                preprocessed_file_path, index=False)
        else:
            pd.DataFrame(preprocessed_data).to_csv(
                preprocessed_file_path, index=False)

        return {"message": "Course added successfully", "course": new_course}
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )


@app.put("/api/py/matakuliah/{course_id}")
async def edit_matakuliah(course_id: str, request: Request):
    try:
        course_id = int(course_id)  # Convert the course_id to an integer
        body = await request.json()
        data_MK = read_data_mk()
        if course_id < 0 or course_id >= len(data_MK):
            raise HTTPException(
                status_code=404,
                detail="Course not found"
            )
        data_MK.loc[course_id, 'namaMK'] = body['nama']
        data_MK.loc[course_id, 'deskripsiMK'] = body['deskripsi']
        write_data_mk(data_MK)

        # Preprocess the updated course data
        preprocessed_data = await get_preprocessed_data(request)
        logger.debug(f"Preprocessed data: {preprocessed_data}")
        if "error" in preprocessed_data:
            raise HTTPException(
                status_code=500,
                detail=f"Error preprocessing data: {preprocessed_data['error']}"
            )

        preprocessed_data_formated = {
            'namaMK': preprocessed_data['nama'][0],
            'deskripsiMK': preprocessed_data['deskripsi'][0]
        }
        # Update the preprocessed data in preprocessed_dataMK.csv
        preprocessed_file_path = './src/api/data/preprocessed_dataMK.csv'
        if os.path.exists(preprocessed_file_path):
            existing_preprocessed_data = pd.read_csv(preprocessed_file_path)
            existing_preprocessed_data.loc[course_id] = preprocessed_data_formated
            existing_preprocessed_data.to_csv(
                preprocessed_file_path, index=False)
        else:
            pd.DataFrame(preprocessed_data_formated).to_csv(
                preprocessed_file_path, index=False)

        return {"message": "Course updated successfully", "course": data_MK.loc[course_id].to_dict()}
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail="Invalid course ID"
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )


@app.delete("/api/py/matakuliah/{course_id}")
async def delete_matakuliah(course_id: str):
    try:
        course_id = int(course_id)  # Convert the course_id to an integer
        data_MK = read_data_mk()
        if course_id < 0 or course_id >= len(data_MK):
            raise HTTPException(
                status_code=404,
                detail="Course not found"
            )
        deleted_course = data_MK.loc[course_id].to_dict()
        data_MK = data_MK.drop(course_id).reset_index(drop=True)
        write_data_mk(data_MK)

        # Remove the corresponding entry from preprocessed_dataMK.csv
        preprocessed_file_path = './src/api/data/preprocessed_dataMK.csv'
        if os.path.exists(preprocessed_file_path):
            existing_preprocessed_data = pd.read_csv(preprocessed_file_path)
            existing_preprocessed_data = existing_preprocessed_data.drop(
                course_id).reset_index(drop=True)
            existing_preprocessed_data.to_csv(
                preprocessed_file_path, index=False)

        return {"message": "Course deleted successfully", "course": deleted_course}
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail="Invalid course ID"
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )


@app.post("/api/py/preprocessing")
async def get_preprocessed_data(request: Request):
    try:
        # If request is a dict, use it directly, otherwise parse JSON
        if hasattr(request, 'body') and isinstance(request.body, dict):
            body = request.body
        else:
            body = await request.json()

        type = body.get('type', 'KM')
        nama = body['nama']
        deskripsi = body['deskripsi']

        # Ensure namaSI and deskripsiSI are lists
        if isinstance(nama, str):
            nama = [nama]
        if isinstance(deskripsi, str):
            deskripsi = [deskripsi]

        # Create dataframe dataSI from namaSI and deskripsi
        data = pd.DataFrame(
            {f'nama{type}': nama, f'deskripsi{type}': deskripsi})

        # Apply preprocessing on the deskripsiSI column
        data[f'deskripsi{type}'] = data[f'nama{type}'].apply(
            clean_text) + ' ' + data[f'deskripsi{type}'].apply(
            clean_text)
        data[f'deskripsi{type}'] = data[f'deskripsi{type}'].apply(
            casefolding_replace)
        data[f'deskripsi{type}'] = data[f'deskripsi{type}'].apply(tokenize)
        data[f'deskripsi{type}'] = data[f'deskripsi{type}'].apply(
            filter_and_translate)
        term_dict = build_term_dict(data[f'deskripsi{type}'])
        data[f'deskripsi{type}'] = apply_stemming(
            data[f'deskripsi{type}'],
            term_dict
        )
        # data[f'deskripsi{type}'] = data[f'deskripsi{type}'].apply(
        #     lambda tokens: [stemmed_wrapper(token) for token in tokens])
        data[f'deskripsi{type}'] = data[f'deskripsi{type}'].apply(
            lemmatize_text)
        data[f'deskripsi{type}'] = data[f'deskripsi{type}'].apply(ubah)
        data[f'deskripsi{type}'] = data[f'deskripsi{type}'].apply(
            filter_stopwords)

        if type == 'SI':
            data.to_csv(
                f'./src/api/data/preprocessed_data{type}.csv', index=False)
        return {"nama": data[f'nama{type}'].tolist(), "deskripsi": data[f'deskripsi{type}'].tolist()}
    except Exception as e:
        logging.error(f"Error processing text: {e}")
        return {"error": str(e)}


class LDARequest(BaseModel):
    num_topics: int = 1
    nama: List[str]
    deskripsi: List[List[str]]
    alpha: str = 'symmetric'
    beta: float = 0.8
    top_n_words: int = 20


@app.post("/api/py/lda_topic")
async def get_lda_topic(request: LDARequest):
    try:
        # Extract input data
        num_topics = request.num_topics
        nama = request.nama
        deskripsi = request.deskripsi
        alpha = request.alpha
        beta = request.beta
        top_n_words = request.top_n_words

        dict_MK, dict_SI, corpus_MK, corpus_SI, dataMK_texts, dataSI_texts = preprocess_data_to_bow(
            deskripsi)

        # Process SI data
        topic_distributions_SI = []
        topic_words_SI = []
        for i, doc_bow in enumerate(corpus_SI):
            try:
                lda_model_SI, topic_words, topic_distributions_SI = evaluate_LDA(
                    doc_bow=doc_bow,
                    dictionary=dict_SI,
                    num_topics=1,
                    text=dataSI_texts[i],
                    alpha=alpha,
                    beta=beta,
                    top_n_words=top_n_words,
                )
                lda_model_SI.save(f'./src/api/model/model_SI.gensim')
                logger.debug(f"topic_words_SI: {topic_words}")
                for words in topic_words:
                    print(f"{', '.join(words)}")
                    topic_words_SI.append(words[:20])
            except Exception as e:
                logger.error(f"Error in processing SI data: {str(e)}")
                raise HTTPException(
                    status_code=500, detail=f"Error in processing SI data: {str(e)}")

        # Process MK data
        topic_distributions_MK = []
        topic_words_MK = []

        for i, doc_bow in enumerate(corpus_MK):
            try:
                lda_model_MK, topic_words, topic_distribution = evaluate_LDA(
                    doc_bow=doc_bow,
                    dictionary=dict_MK,
                    num_topics=1,
                    text=dataMK_texts[i],
                    alpha=alpha,
                    beta=beta,
                    top_n_words=top_n_words
                )

                # Save each LDA model
                lda_model_MK.save(f'./src/api/model/model_MK_{i}.gensim')

                # Store results
                topic_distributions_MK.append(
                    [float(prob) for prob in topic_distribution])
                # print(f"MK {i + 1}:")
                # print("Words for the Topic:")
                for words in topic_words:
                    # print(f"{', '.join(words)}")
                    topic_words_MK.append(words[:20])
                # print("Document-Topic Distribution (Theta):", topic_distribution)
                # topic_words_MK.append(topic_words)
            except Exception as e:
                logger.error(
                    f"Error in processing MK data for document {i}: {str(e)}")
                raise HTTPException(
                    status_code=500, detail=f"Error in processing MK data for document {i}: {str(e)}")

        dataMK = pd.read_csv('./src/api/data/preprocessed_dataMK.csv')
        dataMK = dataMK['deskripsiMK']

        topic_distribution_MK = [[] for _ in range(len(topic_words_MK))]
        # Distribusi Topik MK
        for k in range(len(dataMK)):
            document = dataMK[k]  # Ambil dokumen ke-d
            # logger.debug(f"-- Document ({type(document)}): {document}\n")
            # logger.debug(
            #     f"-- Topic_words_MK ({type(topic_words_MK)}): {topic_words_MK}\n")
            for d in range(len(topic_words_MK)):
                topic_words = topic_words_MK[d]
                # Total kata topik yang ada di dokumen
                n_d_k = sum(document.count(word) for word in topic_words)
                N_d = len(document)  # Total kata dalam dokumen
                # Hitung theta
                theta_d_k = n_d_k / N_d if N_d > 0 else 0  # Mencegah pembagian dengan nol
                topic_distribution_MK[k].append(theta_d_k)
            # logger.debug(
            #     f"-- Topic_words_SI ({type(topic_words_SI)}): {topic_words_SI}\n")
            for d in range(len(topic_words_SI)):
                topic_words = topic_words_SI[d]
                # Total kata topik yang ada di dokumen
                n_d_k = sum(document.count(word) for word in topic_words)
                N_d = len(document)  # Total kata dalam dokumen
                # Hitung theta
                theta_d_k = n_d_k / N_d if N_d > 0 else 0  # Mencegah pembagian dengan nol
                topic_distribution_MK[k].append(theta_d_k)

        # Distribusi Topik SI
        topic_distribution_SI = [[] for _ in range(len(topic_words_SI))]

        for k in range(len(dataSI_texts)):
            document = dataSI_texts[k]  # Ambil dokumen ke-d

            for d in range(len(topic_words_MK)):
                topic_words = topic_words_MK[d]
                # Total kata topik yang ada di dokumen
                n_d_k = sum(document.count(word) for word in topic_words)
                N_d = len(document)  # Total kata dalam dokumen
                # Hitung theta
                theta_d_k = n_d_k / N_d if N_d > 0 else 0  # Mencegah pembagian dengan nol
                topic_distribution_SI[k].append(theta_d_k)

            for d in range(len(topic_words_SI)):
                topic_words = topic_words_SI[d]
                # Total kata topik yang ada di dokumen
                n_d_k = sum(document.count(word) for word in topic_words)
                N_d = len(document)  # Total kata dalam dokumen
                # Hitung theta
                theta_d_k = n_d_k / N_d if N_d > 0 else 0  # Mencegah pembagian dengan nol
                topic_distribution_SI[k].append(theta_d_k)

        # Normalize topic distributions for MK
        for i in range(len(topic_distribution_MK)):
            total_prob = sum(topic_distribution_MK[i])
            if total_prob > 0:
                topic_distribution_MK[i] = [
                    prob / total_prob for prob in topic_distribution_MK[i]]

        # Normalize topic distributions for SI
        for i in range(len(topic_distribution_SI)):
            total_prob = sum(topic_distribution_SI[i])
            if total_prob > 0:
                topic_distribution_SI[i] = [
                    prob / total_prob for prob in topic_distribution_SI[i]]

        topic_distribution_SI = np.matrix(topic_distribution_SI)
        topic_distribution_MK = np.matrix(topic_distribution_MK)
        logger.debug(
            f"Topic Distributions SI Shape: {topic_distribution_SI.shape}")
        logger.debug(
            f"Topic Distributions MK Shape: {topic_distribution_MK.shape}")
        return {
            "SI_topics": topic_distribution_SI.tolist(),
            "MK_topics": topic_distribution_MK.tolist(),
        }
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@app.post("/api/py/directed_similarity")
async def get_directed_similarity(request: Request):
    try:
        body = await request.json()
        # Validate the presence of required keys
        required_keys = ['SI_topics', 'MK_topics']
        for key in required_keys:
            if key not in body:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing key: {key}"
                )

        topic_distributions_SI = body['SI_topics']
        topic_distributions_MK = body['MK_topics']

        # Validate the data types
        if not isinstance(topic_distributions_SI, list) or not isinstance(topic_distributions_MK, list):
            raise HTTPException(
                status_code=400,
                detail="Invalid data format. Expected lists."
            )

        # Validate that the lists are not empty
        if not topic_distributions_SI or not topic_distributions_MK:
            raise HTTPException(
                status_code=400,
                detail="Empty input lists are not allowed."
            )

        try:
            # Convert to numpy arrays
            topic_distributions_SI = np.array(
                topic_distributions_SI, dtype=float)
            topic_distributions_MK = np.array(
                topic_distributions_MK, dtype=float)

            # Ensure both arrays have the same number of features
            if topic_distributions_SI.shape[-1] != topic_distributions_MK.shape[-1]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Feature dimensions do not match: SI topics has {topic_distributions_SI.shape[-1]} features, MK topics has {topic_distributions_MK.shape[-1]} features. Both must have the same number of features."
                )

        except (ValueError, TypeError):
            raise HTTPException(
                status_code=400,
                detail="Input lists must contain valid numeric data."
            )

        similarity_matrix_AB = calculate_directed_AB(
            topic_distributions_SI,
            topic_distributions_MK
        )
        similarity_matrix_BA = calculate_directed_BA(
            topic_distributions_SI,
            topic_distributions_MK
        )

        return {"similarity_AB": similarity_matrix_AB, "similarity_BA": similarity_matrix_BA}

    except HTTPException as he:
        raise he
    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}")
        raise HTTPException(
            status_code=400,
            detail=str(ve)
        )
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )


@app.post("/api/py/recommendation")
async def recommendation(request: Request):
    try:
        body = await request.json()

        # Validate the presence of required keys
        required_keys = ['similarity_AB', 'similarity_BA']
        for key in required_keys:
            if key not in body:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing key: {key}"
                )

        df_AB = body['similarity_AB']
        df_BA = body['similarity_BA']

        data_MK = pd.read_csv('./src/api/data/preprocessed_dataMK.csv')
        data_SI = pd.read_csv('./src/api/data/preprocessed_dataSI.csv')

        # Convert df_AB and df_BA to pandas DataFrames
        if isinstance(df_AB, dict):
            df_AB = pd.DataFrame(df_AB)
        if isinstance(df_BA, dict):
            df_BA = pd.DataFrame(df_BA)

        df_AB.index = data_MK['namaMK'].values
        df_AB.columns = data_SI['namaSI'].values
        df_BA.index = data_MK['namaMK'].values
        df_BA.columns = data_SI['namaSI'].values

        # Validate the data types
        if not isinstance(df_AB, pd.DataFrame) or not isinstance(df_BA, pd.DataFrame):
            raise HTTPException(
                status_code=400,
                detail="Invalid data format. Expected pandas DataFrames."
            )

        recommendation = extract_direct_recommendations(df_AB, df_BA)

        return {"recommendation": recommendation.to_dict('records')}

    except HTTPException as he:
        raise he
    except ValueError as ve:
        logger.error(f"Invalid input: {str(ve)}")
        raise HTTPException(
            status_code=400,
            detail=str(ve)
        )
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )
