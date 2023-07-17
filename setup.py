
def run_setup():
    get_ipython().run_line_magic('pip', 'install torch')
    get_ipython().run_line_magic('pip', 'install torchaudio')
    get_ipython().run_line_magic('pip', 'install tensorflow')
    get_ipython().run_line_magic('pip', 'install sklearn')
    get_ipython().run_line_magic('pip', 'install scikit-learn')
    get_ipython().run_line_magic('pip', 'install seaborn')
    get_ipython().run_line_magic('pip', 'install datasets')
    get_ipython().run_line_magic('pip', 'install optuna')
    get_ipython().run_line_magic('pip', 'install pytorch-lightning')
    get_ipython().run_line_magic('pip', 'install tqdm')
    get_ipython().run_line_magic('pip', 'install transformers')
    get_ipython().run_line_magic('pip', 'install requests')
    get_ipython().run_line_magic('pip', 'install wandb')
    get_ipython().run_line_magic('pip', 'install datasets')
    get_ipython().run_line_magic('pip', 'install soundfile')
    get_ipython().run_line_magic('pip', 'install accelerate')
    get_ipython().run_line_magic('pip', 'install nvidia-tensorrt')
    get_ipython().run_line_magic('pip', 'install nvidia-pyindex')

def run_download_data():
    get_ipython().system('mkdir -p data')
    get_ipython().system('get -O audio.zip https://www.dropbox.com/s/tlbxkdabow9w03i/audio.zip')
    get_ipython().system('wget -O metadata.zip https://www.dropbox.com/s/gi1iwc3xwwl0a4z/metadata.zip')
    get_ipython().system('unzip audio.zip')
    get_ipython().system('unzip metadata.zip')
    get_ipython().system('mv ./audio/ ./data/audio/')
    get_ipython().system('mv ./metadata/ ./data/metadata/')
    get_ipython().system('rm -rf __MACOSX')
    get_ipython().system('rm -rf audio.zip')
    get_ipython().system('rm -rf metadata.zip')
    generate_data_files()


def generate_data_files():
    users_df = pd.read_csv("data/metadata/users.csv")
    samples_df = pd.read_csv("data/metadata/samples.csv")
    get_ipython.system('mkdir -p data/audio4analysis')

    samples_df['actor_gender'] = None
    samples_df['actor_age'] = None
    samples_df['new_file_name'] = None
    
    for index, sample in samples_df.iterrows():
      actor = sample['actor']
      file_name = sample['file_name']
      emotion_expressed = sample['emotion_expressed']
      age = users_df[users_df['username'] == actor]['age'].values[0]
      gender = users_df[users_df['username'] == actor]['gender'].values[0]
      new_file_name = gender + '____' + actor + '____' + emotion_expressed + '____' + file_name
      samples_df.iloc[index, samples_df.columns.get_loc('actor_age')] = age
      samples_df.iloc[index, samples_df.columns.get_loc('actor_gender')] = gender
      samples_df.iloc[index, samples_df.columns.get_loc('new_file_name')] = new_file_name
      os.system('cp data/audio/' + file_name + ' data/audio4analysis/' + new_file_name)

run_setup()
