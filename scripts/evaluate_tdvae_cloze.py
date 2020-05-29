
import fire

class EvalTdvaeCloze(object):
  """Evaluate TDVAE Cloze"""

  def eval(self, prediction_json: str, output_file: str):
    print(f"Evaluate TDVAE Cloze from {prediction_json}, to {output_file}")

if __name__ == '__main__':
  fire.Fire(EvalTdvaeCloze)