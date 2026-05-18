from sagemaker.train.model_trainer import ModelTrainer, Mode
from sagemaker.train.configs import SourceCode, Compute, InputData
from sagemaker.core.helper.session_helper import Session
from sagemaker.core import image_uris

from dotenv import load_dotenv
import os
import time

load_dotenv()

def train_sm():
    start = time.perf_counter()

    print("[1/5] Initializing SageMaker session...")
    session = Session()
    role = os.getenv("SAGEMAKER_ROLE_ARN")
    region = session.boto_region_name
    print(f"    region={region}, role={role}")

    print("[2/5] Retrieving SageMaker training image...")
    training_image = image_uris.retrieve(
        framework="pytorch",
        region=region,
        version="2.0.0",
        py_version="py310",
        instance_type="ml.p3.2xlarge",
        image_scope="training"
    )
    print(f"    training_image={training_image}")

    print("[3/5] Configuring source code and compute...")
    source_code = SourceCode(
        source_dir="train_src",
        entry_script="train.py",
    )

    compute = Compute(
        instance_type="ml.p3.2xlarge",
        instance_count=1,
    )
    print(f"    compute=({compute.instance_type}, count={compute.instance_count})")

    train_data = InputData(
        channel_name="train",
        data_source="s3://musicbench-audio",
    )

    val_data = InputData(
        channel_name="validation",
        data_source="s3://musicbench-audio",
    )

    metadata_data = InputData(
        channel_name="train_metadata",
        data_source="s3://musicbench-audio/metadata/",
    )

    print(
        f"    source_dir={source_code.source_dir}, entry_script={source_code.entry_script}, "
        f"train channel={train_data.channel_name}, validation channel={val_data.channel_name}, metadata channel={metadata_data.channel_name}"
    )

    model_trainer = ModelTrainer(
        training_image=training_image,
        sagemaker_session=session,
        source_code=source_code,
        role=role,
        compute=compute,
        input_data_config=[train_data, val_data, metadata_data],
        base_job_name="sagemaker-training-job",
        training_mode=Mode.SAGEMAKER_TRAINING_JOB,
    )

    print("[4/5] Creating and submitting training job...")
    start_submit = time.perf_counter()
    model_trainer.train(input_data_config=[train_data, val_data], wait=False, logs=False)
    submit_duration = time.perf_counter() - start_submit
    print(f"    submission complete in {submit_duration:.2f}s")

    total_duration = time.perf_counter() - start
    print("[5/5] Training job submitted")
    print(f"Total script submission time: {total_duration:.2f}s")
