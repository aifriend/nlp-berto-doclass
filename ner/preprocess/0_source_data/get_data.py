import base64
import os
import random
import re

from GbcProductIrphCleaner import GbcProductIrphCleaner
from extract.Helper import Helper, ExtractSection
from extract.Hfile import dump, clean_doc
from services.names.GbcNamingService import GbcNamingService
from services.person.ExtractorService import ExtractorService
from services.person.role.extractor.GbcRoleRepresentationExtractor import GbcRoleRepresentationExtractor

file_id = ""
file_path = "test"

_regex_start = [
    re.compile(r"(do[nñ]a?\.?.{1,100}representacion.{1,15}de.{1,200})",
               re.IGNORECASE | re.MULTILINE | re.DOTALL),
    re.compile(r"(do[nñ]a?\.?.{1,100}nombre.y.representacion.{1,200})",
               re.IGNORECASE | re.MULTILINE | re.DOTALL),
    re.compile(r"(do[nñ]a?.{1,150}nombre.y.representacion.de.{1,15}do[nñ]a?.{1,500})",
               re.IGNORECASE | re.MULTILINE | re.DOTALL),
    re.compile(r"(\)[\W\w]{0,10}do[nñ]a?.{1,150}nombre.y.representacion.de.{1,105}do[nñ]a?.{80,8500})",
               re.IGNORECASE | re.MULTILINE | re.DOTALL),
    re.compile(r"(do[nñ]a?\.?.{,100}apodera.{1,15}de.{1,200})",
               re.IGNORECASE | re.MULTILINE | re.DOTALL),
    re.compile(r"([don].{1,150}apodera.de.{1,15}[don].{1,500})",
               re.IGNORECASE | re.MULTILINE | re.DOTALL),
    re.compile(r"(do[nñ]a?\.?.{1,100}apodera.{1,15}de.{1,500})",
               re.IGNORECASE | re.MULTILINE | re.DOTALL),
    re.compile(r"(do[nñ]a?\.?.{1,100}calidad.de.apodera.{1,200})",
               re.IGNORECASE | re.MULTILINE | re.DOTALL),
    re.compile(r"(.{1,100}representacion.{1,15}de.[^)]{0,200})",
               re.IGNORECASE | re.MULTILINE | re.DOTALL),
    re.compile(r"(.{1,100}representacion,.*?de.{0,200})",
               re.IGNORECASE | re.MULTILINE | re.DOTALL),
    re.compile(r"(.{1,100}nombre.y.representacion.{1,15}de.{0,200})",
               re.IGNORECASE | re.MULTILINE | re.DOTALL),
    re.compile(r"(.{1,100}apodera.de.{1,500})",
               re.IGNORECASE | re.MULTILINE | re.DOTALL),
    re.compile(r"(.{1,100}apodera.{1,15}solidario.{1,200})",
               re.IGNORECASE | re.MULTILINE | re.DOTALL),
]


def _regex_extractor(_data, match_list):
    matchesList = list()
    for reg in _regex_start:
        matches = re.findall(reg, _data)
        matches_list = list(matches)
        for match in matches_list:
            matchesList.append(match)

    if isinstance(match_list, list) and matchesList:
        match_list.append((
            GbcRoleRepresentationExtractor.TOKEN_REPRESENTA, matchesList))

    return match_list


if __name__ == '__main__':
    name_service = GbcNamingService()
    irph_cleaner = GbcProductIrphCleaner()
    path = os.path.dirname(os.path.realpath(__file__)) + f"/data/{file_path}/"
    result_list = Helper.get_path(path)
    if isinstance(result_list, dict):
        total = len(result_list)
        left = total
        keys = list(result_list.keys())
        random.shuffle(keys)
        extract_section = ExtractSection()
        irphCleaner = GbcProductIrphCleaner()
        for file in keys:
            data = result_list[file]
            _, file_name = os.path.split(file)
            if True or file_name == f"{file_id}.pdf.txt":
                try:
                    data_decoded = base64.b64decode(data).decode('utf-8')
                except:
                    data_decoded = data
                data_decoded = ExtractorService.simplify(clean_doc(irph_cleaner.clean(data_decoded, False)))
                try:
                    # text extraction
                    role_list = list()
                    regex_list = _regex_extractor(data_decoded, role_list)
                    if regex_list:
                        role_list.extend(regex_list)

                    # save to file
                    _, fn = os.path.split(file)
                    if len(role_list):
                        for reg_rep in role_list:
                            for reg in reg_rep[1]:
                                dump(f"{reg}", "/data/reg_representation.txt")
                        print(f"FOUND: {fn}")
                    else:
                        dump(f"{fn}", "/data/reg_rep_not_found.txt")
                        print(f"NOT FOUND: {fn}")

                except Exception as exc:
                    dump(f"{file}", "/data/reg_rep_exception.txt")
                    print(f"EXCEPTION: {file}")

            left = left - 1
            print(f"LEFT: {left}")
