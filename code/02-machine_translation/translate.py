## Setup
import os
import sys

import pandas as pd
import numpy as np

from tqdm.auto import tqdm

import torch
import gc

import iso639
import html

import easynmt
from google.oauth2 import service_account
from google.cloud import translate_v2 as gt
import deepl

from typing import Union, List, Callable

import time
from datetime import datetime

### NLTK

import nltk.data
from nltk.tokenize import sent_tokenize

resources_dir = os.path.join(nltk.data.path[0], 'tokenizers', 'punkt')

if not os.path.exists(resources_dir):
    import nltk
    nltk.download('punkt', quiet=True)

NLTK_LANGS = [f.split('.')[0] for f in os.listdir(resources_dir) if f.endswith('.pickle')]
NLTK_LANG_CODE2NAME = {iso639.to_iso639_1(l): l for l in NLTK_LANGS}


### custom functions
ts = lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S")
log = lambda x: print(f'[{ts()}] {x}')

# chunk list of sentences into smaller chunks
def chunk(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


def clean_memory(device):
    if 'cuda' in str(device):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    elif 'mps' in str(device):
        torch.mps.empty_cache()
    else:
        pass
    gc.collect()


def translate_batch_safely(texts: list, translation_fun: Callable, device: Union[str, torch.device], **kwargs) -> list:
    """
    Translates a batch of texts using the model, handling potential errors.

    Parameters:
        texts (list): A list of texts to be translated.
        translation_fun (Callable): The translation function to be used.
        **kwargs: Additional keyword arguments to be passed to `translation_fun`.

    Returns:
        list: A list of translated texts.
    """
    try:
        # Attempt to translate the batch of texts using the model
        res = translation_fun(texts, **kwargs)
    except Exception as e:
        # If the exception is _not_ related to running out of memory, ...
        if 'out of memory' not in str(e):
            # ... raise the exception
            raise e
        # but if the error was due running out of memory, ...
        else:
            args = dict(**kwargs)
            if 'batch_size' in args:
                del args['batch_size']
            
            clean_memory(device)
            res = [None] * len(texts)
            # ... try translating each text individually
            for i, text in enumerate(texts):
                try:
                    res[i] = translation_fun(text, batch_size=1, **args)
                except Exception as e:
                    msg = f'WARNING: couldn\'t translate text "{text[:min(50, len(text))]}". Reason: {str(e)}'
                    # If unable to translate a text, try to translate sentence by sentence
                    if 'source_lang' not in args or args['source_lang'] not in NLTK_LANGS:
                        log(msg)
                        continue
                    sents = sent_tokenize(text.strip(), language=NLTK_LANG_CODE2NAME[args['source_lang']])
                    try:
                        sents = [translation_fun(sent, batch_size=1, **args) for sent in sents]
                    except:
                        log(msg)
                        continue
                    else:
                        res[i] = ' '.join(sents)
    return res


def translate_in_batches(
        texts: list, 
        batch_size: int, 
        max_chars_per_minute: Union[None,int]=None,
        verbose: bool=False, 
        pbar_desc: Union[None,str]=None, 
        **kwargs
    ) -> list:
    """
    Translates a list of texts in batches using the `translate_batch_safely` function.

    Parameters:
        texts (list): A list of texts to be translated.
        batch_size (int): The size of each translation batch.
        max_chars_per_minute (int, optional): Maximum number of characters to translate (can be used to respect API rate limits)
        verbose (bool): Whether to print messages and a progress bar
        pbard_desc (str, optional): The description of the progress bar
        **kwargs: Additional keyword arguments to be passed to the `translate_batch_safely` function.

    Returns:
        list: A list of translated texts.
    """
    # Initialize an empty list to store the translations
    translations = []
    n_batches = len(texts)//batch_size
    if verbose:
        pbar = tqdm(total=n_batches, desc=pbar_desc)
    # for tracking translated characters per minute
    n_chars = 0
    st = time.time()
    # Iterate over the batches of texts
    for batch in chunk(texts, batch_size):
        # main task: translate the batch of texts
        translations += translate_batch_safely(batch, **kwargs)
        if verbose: 
            pbar.update(1)
        # additional logic if max characters/minute limit set
        if max_chars_per_minute is not None:
            # update number of translated characters
            n_chars += sum(map(len, batch))
            # if max character not yet reached, continue 
            if n_chars < max_chars_per_minute:
                continue
            # else
            else:
                # compute how many seconds have elapsed since start/last pause
                elapsed = (time.time()-st)
                # if necessary, pause to complete the minute
                if elapsed < 60:
                    time.sleep(60-elapsed)
                # reset counters
                n_chars = 0
                st = time.time()
    if verbose: 
        pbar.close()
    return translations


# helpers
def is_string_series(s: pd.Series):
    """
    Test if pandas series is a string series/series of strings
    
    source: https://stackoverflow.com/a/67001213
    """
    if isinstance(s.dtype, pd.StringDtype):
        # The series was explicitly created as a string series (Pandas>=1.0.0)
        return True
    elif s.dtype == 'object':
        # Object series, check each value
        return all(isinstance(v, str) or (v is None) or (np.isnan(v)) for v in s)
    else:
        return False

def is_nonempty_string(s: pd.Series):
    return np.array([isinstance(v, str) and len(v) > 0 for v in s], dtype=bool)

def translate_df(
        df: pd.DataFrame, 
        translation_function: Callable,
        supported_languages: List[str],
        text_col: str = 'text', 
        lang_col: str = 'lang',
        target_language: str = 'en',
        target_col: str = 'translation',
        overwrite_target_col: bool = False,
        device: Union[str, torch.device] = 'cpu',
        batch_size: int = 16,
        verbose: bool = False,
        **kwargs
    ):
    """
    Translates the texts in a data frame from the source languages specified in a column to a target language and add the translations to the data frame.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the texts to be translated.
        translation_function (Callable): The translation function to be used.
        supported_languages (List[str]): A list language codes supported by the translation model.
        text_col (str): The name of the column in the DataFrame that contains the texts to be translated. Default is 'text'.
        lang_col (str): The name of the column in the DataFrame that contains the language codes. Default is 'lang'.
        target_language (str): The target language to translate the texts to. Can be either an ISO 639-1 or ISO 639-2 language code. Default is 'en'.
        target_col (str): The name of the column in the DataFrame to store the translations. Default is 'translation'.
        supported_languages (List[str]): A list of ISO 639-1 or ISO 639-2 language codes supported by the translation model. Default is None.
        device (Union[str, torch.device]): The device to use for translation. Default is 'cpu' but should be compatible with device used by translation model.
        batch_size (int): The size of each translation batch. Default is 16.
        **kwargs: Additional keyword arguments to be passed to the `translate_in_batches` function which, in turn, passes them to the translation function.

    Returns:
        pd.DataFrame: The DataFrame with the translated texts in column `target_col` in the target language `target_lang`.
    """
    # validate the inputs
    assert text_col in df.columns, f'Column "{text_col}" not found in data frame.'
    assert is_string_series(df[text_col]), f'Column "{text_col}" is not a series of string values.'
    assert lang_col in df.columns, f'Column "{lang_col}" not found in data frame.'
    assert is_string_series(df[lang_col]), f'Column "{lang_col}" is not a series of string values.'
    assert target_language is not None, 'Target language must be specified.'
    if not overwrite_target_col:
        assert target_col not in df.columns, f'Column "{target_col}" already exists in data frame.'
    assert translation_function is not None, 'Translation function must be specified.'
    assert batch_size > 0, 'Batch size must be greater than 0.'
    assert supported_languages is not None, 'Supported languages must be specified.'
    assert isinstance(supported_languages, list), 'Supported languages must be a list.'
    assert len(supported_languages) > 0, 'Supported languages must not be empty.'
    assert all([isinstance(l, str) for l in supported_languages]), 'Supported languages must be a list of strings.'
    
    # check whether the model supports the target language
    langs = df['lang'].unique().tolist()
    # try to get the ISO 639-1 or ISO 639-2 language code for each language in the data frame
    langs_map = {
        l: l if iso639.is_valid639_1(l) else iso639.to_iso639_1(l) if iso639.is_valid639_2(l) else None 
        for l in langs
    }
    # check whether there are unsupported languages
    not_supported = [
        l 
        for l, c in langs_map.items() 
        if l not in supported_languages and c not in supported_languages and l != target_language and c != target_language
    ]
    # print warning message if there are unsupported languages
    if len(not_supported) > 0:
        log(
            f'WARNING: values {not_supported} in column "{lang_col}" are not supported by NMT model. Texts with these values will not be translated.'
        )
    # now update language mapping with "correct" language codes (use ISO code if available, otherwise use original indicator from the data frame)
    langs_map = {
        l: c if c in supported_languages else l if l in supported_languages else None 
        for l, c in langs_map.items()
    }

    # create new column for translation
    df[target_col] = [None]*len(df)

    # iterate over languages
    for l, d in df.groupby(lang_col):
        lang_code = langs_map[l]
        # just copy texts if source language is the target language
        if lang_code == target_language or l == target_language:
            df.loc[d.index, target_col] = d[text_col].tolist()
            continue
        if '-' in target_language:
            # Deepl specific for target langs en-GB and en-US
            tmp = target_language.split('-')[0].strip()
            if lang_code == tmp or l == tmp:
                df.loc[d.index, target_col] = d[text_col].tolist()
                continue
        # skip unsupported languages
        if l in not_supported or lang_code is None:
            continue
        # test for each text value if non-empty string
        flag = is_nonempty_string(d[text_col])
        if any(~flag):
            log(f'WARNING: {sum(~flag)} empty or non-string text(s) in "{l}"')
        df.loc[d.index[flag], target_col] = translate_in_batches(
            texts=d[text_col][flag].tolist(), # <== only translate non-empty texts
            translation_fun=translation_function,
            device=device,
            batch_size=batch_size, 
            source_lang=lang_code, 
            target_lang=target_language,
            verbose=verbose, 
            pbar_desc=f'translating {len(d)} text(s) from "{l}"',
            **kwargs
        )
    
    return df


def translate_df_with_easynmt(df, args):
    """
    Translates a DataFrame using the EasyNMT model.

    Args:
        df (pandas.DataFrame): The DataFrame to be translated.
        args: Additional arguments for the translation process.

    Returns:
        pandas.DataFrame: The translated DataFrame.
    """

    try:
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        model = easynmt.EasyNMT(args.model_name, device=device)
        multi_gpu = device=='cuda' and (n_gpus:=torch.cuda.device_count()) > 1
        if multi_gpu:
            devs = 'cuda:'+','.join([str(i) for i in range(torch.cuda.device_count())])
            log(f'Using device "{devs}"')
            process_pool = model.start_multi_process_pool()
        else:
            log(f'Using device "{model.device}"')
    except Exception as e:
        log(f'WARNING: could not load model "{args.model_name}"')
        raise e
    
    tgt_lang = [l.lower() for l in model.get_languages() if args.target_language.lower() == l.lower() or args.target_language.lower() in l.lower()]
    if len(tgt_lang) == 0:
        raise ValueError(f'Target language "{args.target_language}" not supported by DeepL.')
    if len(tgt_lang) > 1:
        raise ValueError(f'Target language "{args.target_language}" ambiguous. Please specify one of {tgt_lang}.')
    tgt_lang = tgt_lang[0]
    src_langs = model.get_languages(target_lang = tgt_lang)

    df = df.copy(deep=True)
    
    if multi_gpu:
        translate_fun = lambda **kwargs: model.translate_multi_process(process_pool, **kwargs)
    else:
        translate_fun = model.translate 
    
    try:
        df = translate_df(
            df=df, 
            # data frame arguments
            text_col=args.text_col,
            lang_col=args.lang_col,
            target_language=tgt_lang, 
            target_col=args.target_col if hasattr(args, 'target_col') else f'{args.text_col}_mt_{args.model_name.lower()}',
            overwrite_target_col=args.overwrite_target_column if hasattr(args, 'overwrite_target_column') else False,
            # translation model arguments
            translation_function=translate_fun,
            supported_languages=src_langs,
            batch_size=args.batch_size if hasattr(args, 'batch_size') else 16,
            device=model.device,
            # arguments forwarded to translation function
            beam_size=5,
            perform_sentence_splitting=args.split_sentences if hasattr(args, 'split_sentences') else False,
            show_progress_bar=False, 
            # print progress bar
            verbose=args.verbose,
        )
    except Exception as e:
        log(f'WARNING: Error during translation "{str(e)}". Returning data frame with translations so far.')
    
    return df


def translate_df_with_google(df, args):
    # get API key
    try:
        credentials = service_account.Credentials.from_service_account_file(args.api_key_file)
    except Exception as e:
        raise ValueError(f'Could not load API key file "{args.api_key_file}". Reason: {str(e)}')
    
    # initialize a `translator` instance
    try:
        translator = gt.Client(credentials=credentials)
    except Exception as e:
        raise ValueError(f'Could not connect to Google Cloud Translation API. Reason: {str(e)}')
    
    # get source and target languages
    src_langs = [l['language']  for l in translator.get_languages()]
    tgt_lang = [l.lower() for l in src_langs if args.target_language.lower() == l.lower() or args.target_language.lower() in l.lower()]
    if len(tgt_lang) == 0:
        raise ValueError(f'Target language "{args.target_language}" not supported by Google Translate.')
    if len(tgt_lang) > 1:
        raise ValueError(f'Target language "{args.target_language}" ambiguous. Please specify one of {tgt_lang}.')
    tgt_lang = tgt_lang[0]
    
    df = df.copy(deep=True)
    tgt_col = f'{args.text_col}_mt_google'
    
    def translate_util(values, target_lang, source_lang, **kwargs):
        return translator.translate(values=values, target_language=target_lang, source_language=source_lang, **kwargs)

    # translate
    try:
        df = translate_df(
            df=df, 
            # data frame arguments
            text_col=args.text_col,
            lang_col=args.lang_col,
            target_language=tgt_lang,
            target_col=args.target_col if hasattr(args, 'target_col') else tgt_col,
            overwrite_target_col=args.overwrite_target_column if hasattr(args, 'overwrite_target_column') else False,
            # translation model arguments
            translation_function=translate_util,
            supported_languages=src_langs,
            batch_size=args.batch_size if hasattr(args, 'batch_size') else 128,
            max_chars_per_minute=6_000_000, # <== Google Cloud Translation API rate limit for "v2 and v3 general model characters per minute"
            # print progress bar
            verbose=args.verbose,
        )
    except Exception as e:
        log(f'WARNING: Error during translation "{str(e)}". Returning data frame with translations so far.')
    
    try:
        # post-process translation result
        df[tgt_col] = df[tgt_col].apply(lambda x: x if isinstance(x, str) else x['translatedText'] if x is not None else None)
        # replace HTML entities
        flag = df[tgt_col].str.contains(r'&[^; ]+;')
        df.loc[flag, tgt_col] = df.loc[flag, tgt_col].apply(html.unescape)
    except Exception as e:
        log(f'WARNING: Error during post-processing "{str(e)}". Returning data frame with translations so far.')
    
    return df


def translate_df_with_deepl(df, args):
    # get API key
    try:
        with open(args.api_key_file) as f:
            api_key = f.read().strip()
    except Exception as e:
        raise ValueError(f'Could not load API key file "{args.api_key_file}". Reason: {str(e)}')
        
    # initialize a `Translator` instance
    try:
        translator = deepl.Translator(api_key)
    except Exception as e:
        raise ValueError(f'Could not connect to DeepL API. Reason: {str(e)}')

    # get source and target languages
    src_langs = [l.code.lower() for l in translator.get_source_languages()]
    tgt_lang = [l.code.lower() for l in translator.get_target_languages() if args.target_language.lower() == l.code.lower() or args.target_language.lower() in l.code.lower()]
    if len(tgt_lang) == 0:
        raise ValueError(f'Target language "{args.target_language}" not supported by DeepL.')
    if len(tgt_lang) > 1:
        raise ValueError(f'Target language "{args.target_language}" ambiguous. Please specify one of {tgt_lang}.')
    tgt_lang = tgt_lang[0]
    
    df = df.copy(deep=True)
    tgt_col = f'{args.text_col}_mt_deepl'
    
    # translate
    try:
        df = translate_df(
            df=df, 
            # data frame arguments
            text_col=args.text_col,
            lang_col=args.lang_col,
            target_language=tgt_lang,
            target_col=args.target_col if hasattr(args, 'target_col') else tgt_col,
            overwrite_target_col=args.overwrite_target_column if hasattr(args, 'overwrite_target_column') else False,
            # translation model arguments
            translation_function=translator.translate_text,
            supported_languages=src_langs,
            batch_size=args.batch_size if hasattr(args, 'batch_size') else 1280,
            # arguments forwarded to translator.translate_text()
            split_sentences=0 if not hasattr(args, 'split_sentences') else 1 if args.split_sentences else 0,
            # print progress bar
            verbose=args.verbose,
        )
    except Exception as e:
        log(f'WARNING: Error during translation "{str(e)}". Returning data frame with translations so far.')
    
    try:
        # post-process translation result
        df[tgt_col] = df[tgt_col].apply(lambda x: x if isinstance(x, str) else x.text if x is not None else None)
    except Exception as e:
        log(f'WARNING: Error during post-processing "{str(e)}". Returning data frame with translations so far.')
    
    return df

def main(args):

    if args.verbose:
        log(f'Reading input file "{args.input_file}"')
    
    # get the file extension
    ext = os.path.splitext(args.input_file)[1]
    sep = None
    match ext.lower():
        case '.csv': sep = ','
        case _: sep = '\t'
    
    # read the file
    try:
        df = pd.read_csv(args.input_file, sep=sep)
    except Exception as e:
        raise ValueError(f'Could not read input file "{args.input_file}". Reason: {str(e)}')

    assert args.text_col in df.columns, ValueError(f'--text_col column "{args.text_col}" not found in data frame.')
    assert args.lang_col in df.columns, ValueError(f'--lang_col column "{args.lang_col}" not found in data frame.')
    
    if args.test:
        if args.verbose: log(f'Running in test mode. Using {args.test_n} random samples from each language.')
        df = df.groupby(args.lang_col).sample(args.test_n, random_state=1234, replace=True).drop_duplicates().sample(frac=1.0, random_state=1234).reset_index(drop=True)

    # translate
    if args.translator == 'easynmt':
        if args.verbose:
            log(f'Translating texts with easynmt (model "{args.model_name}")')
        df = translate_df_with_easynmt(df, args)
    elif args.translator == 'google':
        if args.verbose:
            log(f'Translating texts with Google Translate')
        df = translate_df_with_google(df, args)
    elif args.translator == 'deepl':
        if args.verbose:
            log(f'Translating texts with DeepL')
        df = translate_df_with_deepl(df, args)
    else:
        raise ValueError(f'Translator "{args.translator}" not supported. Please specify one of {allowed_translators}.')
    
    # write output
    fp = args.output_file if args.output_file else args.input_file
    ext = os.path.splitext(fp)[1]
    match ext.lower():
        case '.csv': sep = ','
        case _: sep = '\t'
    log(f'Writing translated data to "{fp}"')
    try:
        df.to_csv(fp, sep=sep, index=False, encoding='utf-8')
    except Exception as e:
        raise ValueError(f'Could not write output file "{fp}". Reason: {str(e)}')

if __name__ == '__main__':

    import argparse

    desc = [
        'Translate texts in tabular data file',
        '',
        'This script allows to use easyNMT or the Google Cloud Translation or DeepL APIs for machine translation.',
    ]
    parser = argparse.ArgumentParser(description='\n'.join(desc), formatter_class=argparse.RawTextHelpFormatter)
    
    # arguments
    parser.add_argument('-i', '--input_file', type=str, help='Input file', required=True)
    parser.add_argument('-o', '--output_file', type=str, default=None, help='Output file. If not specified and --overwrite_output_file is set, input file will be overwritten.')
    parser.add_argument('--overwrite_output_file', action='store_true', help='Overwrite output file if it exists. If output file not specified, input file will be overwritten.')
    parser.add_argument('--text_col', type=str, default='text', help='Name of column containing texts to be translated')
    parser.add_argument('--lang_col', type=str, default='lang', help='Name of column that indicates to-be-translated texts\' language codes')
    # note: could add target_col argument to allow for manual naming of column that will record translations
    parser.add_argument('--target_language', type=str, default='en', help='ISO 639-1 language code of "target" language to translate texts to')
    parser.add_argument('--overwrite_target_column', action='store_true', help='Overwrite target column if it exists.')
    parser.add_argument('--translator', type=str, default='easynmt', help='Translator to use. One of "easynmt", "google", or "deepl".')
    parser.add_argument('--model_name', type=str, default='m2m_100_418M',  help='Name of model to use for translation. Only used if translator is "easynmt". See https://github.com/UKPLab/EasyNMT#available-models for available models.') 
    parser.add_argument('--api_key_file', type=str, default=None, help='Path to file containing API key. Only used if translator is "google" or "deepl".')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for translation. Recommended to set to 32 or lower if translator "easynmt" is used with GPU.')
    parser.add_argument('--split_sentences', action='store_true', help='Split sentences before translation. CAUTION: Sentence splitting in supported models is punctuation based and might be wrong, potentially impairing translation quality.')
    parser.add_argument('--verbose', action='store_true', help='Print progress bar and other messages.')
    parser.add_argument('--test', action='store_true', help='Run in test mode.')
    parser.add_argument('--test_n', type=int, default=32, help='number of texts to sample per language in test mode')

    args = parser.parse_args()

    # validate inputs
    
    # check that input_file exists
    assert os.path.exists(args.input_file), ValueError(f'Input file "{args.input_file}" not found.')
    # check that input_file is a file
    assert os.path.isfile(args.input_file), ValueError(f'Input file "{args.input_file}" is not a file.')
    # check that input_file is a csv, tsv, or tab file
    allowed_extensions = ['.csv', '.tsv', '.tab']
    assert any(args.input_file.lower().endswith(ext) for ext in allowed_extensions), ValueError(f'Input file "{args.input_file}" is not a csv, tsv, or tab file.')
    
    # check output_file
    if args.output_file is not None:
        assert not os.path.exists(args.output_file) or args.overwrite_output_file, ValueError(f'Output file "{args.output_file}" already exists. Use --overwrite_output_file to overwrite.')
    else:
        assert args.overwrite_output_file, ValueError(f'No output file specified. Set --overwrite_output_file to overwrite input file.')
    
    # check that translator is supported
    allowed_translators = ['easynmt', 'google', 'deepl']
    args.translator = args.translator.lower()
    assert args.translator in allowed_translators, ValueError(f'Translator "{args.translator}" not supported. Please specify one of {allowed_translators}.')

    if args.translator == 'easynmt':
        # ensure that model_name is specified if translator is easynmt
        assert args.model_name is not None, ValueError(f'For translator "{args.translator}", --model_name must be specified.')
        if args.batch_size > 32 and (torch.cuda.is_available() or torch.backends.mps.is_available()):
            log(f'WARNING: batch size {args.batch_size} might lead to memory overflow for model "{args.model_name}".')
    else:
        assert args.api_key_file is not None, ValueError(f'For translator "{args.translator}", --api_key_file must be specified.')
        assert os.path.exists(args.api_key_file), ValueError(f'API key file "{args.api_key_file}" not found.') 
        assert os.path.isfile(args.api_key_file), ValueError(f'API key file "{args.api_key_file}" is not a file.') 

    # run main function
    try:
        main(args)
    except Exception as e:
        log(f'ERROR: {str(e)}')
        sys.exit(1)
    else:
        sys.exit(0)
