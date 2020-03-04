''' Simple wrapper script to extract the existing SQL Lite database to a format for th new predictor.
'''

import argparse
import dataset
import pandas
import tqdm
from jsonlines import jsonlines

engine_kwargs = {"pool_recycle": 3600, "connect_args": {'timeout': 300, "check_same_thread": False}}

def add_coreference_features(args):
    print(args)

    database = args["database"]
    dataset_db = f"sqlite:///{database}"


    story_ids_to_predict = []
    if "include_list" in args:
        story_id_df = pandas.read_csv(args["include_list"], engine='python')
        story_ids_to_predict = set(story_id_df['story_id'])

    with jsonlines.open(args['output_json'], mode='w') as writer:
        with dataset.connect(dataset_db, engine_kwargs=engine_kwargs) as db:
            stories = db.query('SELECT * FROM story')

            for story in stories:

                story_id = story["id"]

                if story_id in story_ids_to_predict:

                    story_json_dict = {}
                    story_json_dict["story_id"] = story_id

                    sentences = [s for s in db.query(
                        f'SELECT * FROM sentence INNER JOIN sentence_lang on sentence.id = sentence_lang.sentence_id '
                        f'WHERE sentence.story_id = {story_id} and sentence_lang.lang = "en" '
                        f'and sentence_lang.nonsense = false and sentence_lang.ascii_chars=true and sentence.sentence_len >= 2 ORDER BY id')]

                    extracted_sentences = []
                    for s in sentences:
                        ext_map = {}
                        for f in ["sentence_num","text"]:
                            ext_map[f] = s[f]
                        extracted_sentences.append(ext_map)

                    story_json_dict["sentences"] = extracted_sentences

                    if len(extracted_sentences) > 0:
                        writer.write(story_json_dict)


parser = argparse.ArgumentParser(
    description='Extract text to the predictor format used in the new ')
parser.add_argument('--database', required=True, type=str, help="The SQLLite datase to read the stories from.")
parser.add_argument('--include-list', required=False, type=str, help="The SQLLite database to read the stories from.")
parser.add_argument('--output-json', type=str, required=True, help="The location to save the output json files.")

args = parser.parse_args()

add_coreference_features(vars(args))