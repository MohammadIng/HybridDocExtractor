from src.data_tranformation.data_trasnformer import DataTransformer


def run_processing_document(image_name="form_3_1A.png"):
    data_transformer = DataTransformer(
                                    image_name=image_name,
                                    rerun_ocr=False)
    # data_transformer.save_data_into_db()
    # data_transformer.create_forms_table_if_not_exists()

run_processing_document()