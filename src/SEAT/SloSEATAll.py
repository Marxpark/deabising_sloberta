import json
import re

import torch.nn.functional as F

moski_poklici_f = open("../data/sorted_moski.txt")
zenski_poklici_f = open("../data/sorted_zenski.txt")
#  i removed metalurginja
moski_poklici = moski_poklici_f.readlines()
zenski_poklici = zenski_poklici_f.readlines()

moski_besede_f = open("../data/moske-SEAT.txt")
zenski_besede_f = open("../data/zenske-SEAT.txt")

moski_stavki = moski_besede_f.readlines()
zenski_stavki = zenski_besede_f.readline()

nevtralni_poklici_f = open("../data/nevtralni_poklici.txt")
nevtralni_stavki = nevtralni_poklici_f.readlines()

moski_besede = open("../data/moske.txt")
moske_besede = moski_besede.readlines()
zenski_besede = open("../data/zenske.txt")
zenske_besede = zenski_besede.readlines()


from transformers import AutoTokenizer, AutoModel
import statistics

import torch

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


def get_stats(list_of_results):
    average = statistics.mean(list_of_results)
    std_dev = statistics.stdev(list_of_results)
    min_value = min(list_of_results)
    max_value = max(list_of_results)
    min_index = list_of_results.index(min_value)
    max_index = list_of_results.index(max_value)
    report = {
        "average": average,
        "standard_deviation": std_dev,
        "min_value": min_value,
        "min_sentence": nevtralni_stavki[min_index],
        "max_value": max_value,
        "max_sentence": nevtralni_stavki[max_index]

    }
    return report


def extract_path(full_path):
    # Use regex to extract the value between "param_search/" and "/checkpoint-best"
    match = re.search(r'/param_search/(.*?)/checkpoint-best', full_path)
    if match:
        return match.group(1)  # The captured path part
    return "SLOBERTAORIGINAL"  # If no match is found


# from Moški je oseba. to pirotehnik je oseba
def run_seat(path_to_model="EMBEDDIA/sloberta", save_path="SLOBERTA"):
    #### SEAT
    all_seat_comparisons = []
    skupni_moski_avg = 0
    skupni_zenski_avg = 0
    seat_results = ""
    tokenizer = AutoTokenizer.from_pretrained(path_to_model)
    model = AutoModel.from_pretrained(path_to_model)
    model = model.to(device)

    print("Starting SEAT")
    for i in range(len(nevtralni_stavki)):
        # 'administrativno delo'
        current_target_sentence = torch.tensor(tokenizer.encode(nevtralni_stavki[i])).unsqueeze(0).to(device)

        moski_average = 0
        zenski_average = 0
        print("Evaluating sentance {}".format(nevtralni_stavki[i]))
        with torch.no_grad():
            current_target_sentence = model(current_target_sentence)[0]
        current_target_sentence = torch.sum(current_target_sentence[-1], dim=0).unsqueeze(0)

        for index_stavka in range(len(moski_stavki)):
            #  moški je oseba
            current_source_moski = torch.tensor(tokenizer.encode(moski_stavki[index_stavka])).unsqueeze(0).to(device)
            #  ženska je oseba
            current_source_zenski = torch.tensor(tokenizer.encode(zenski_stavki[index_stavka])).unsqueeze(0).to(device)

            with torch.no_grad():
                current_source_moski = model(current_source_moski)[0]
                current_source_zenski = model(current_source_zenski)[0]

            current_source_moski = torch.sum(current_source_moski[-1], dim=0).unsqueeze(0)
            current_source_zenski = torch.sum(current_source_zenski[-1], dim=0).unsqueeze(0)

            # current_target_sentence = F.normalize(current_target_sentence, p=2, dim=-1)
            # current_source_moski = F.normalize(current_source_moski, p=2, dim=-1)
            # current_source_zenski = F.normalize(current_source_zenski, p=2, dim=-1)

            moski_cos_sim = F.cosine_similarity(current_source_moski, current_target_sentence).item()
            zenski_cos_sim = F.cosine_similarity(current_source_zenski, current_target_sentence).item()

            moski_average += moski_cos_sim
            zenski_average += zenski_cos_sim

        # povprečje vseh spolno zaznamovanih stavkov do tarčenga dela s poklicom
        moski_average /= len(moski_stavki)
        zenski_average /= len(moski_stavki)

        skupni_moski_avg += moski_average
        skupni_zenski_avg += zenski_average
        seat_results += "Nevtralni stavek: {}, mosko povpreje: {}, zensko povprecje {}, razlika {} \n".format(
            nevtralni_stavki[i], moski_average, zenski_average, moski_average - zenski_average
        )
        all_seat_comparisons.append((moski_average - zenski_average))
    skupni_moski_avg /= len(nevtralni_stavki)
    skupni_zenski_avg /= len(nevtralni_stavki)
    seat_results += "Skupni moski average: {} - zenski: {} - razlika: {}".format(skupni_moski_avg, skupni_zenski_avg,
                                                                                 skupni_moski_avg - skupni_zenski_avg)
    seat_res = open("seat_res.txt", "w")
    seat_res.write(seat_results)
    seat_res.close()

    report = get_stats(all_seat_comparisons)
    return report


#### heilman seat
moska_imena = [
    "Franc",
    "Janez",
    "Marko",
    "Ivan",
    "Anton",
    "Andrej",
    "Jožef",
    "Luka",
    "Matej",
    "Jože",
    "Peter",
    "Tomaž",
    "Marjan",
    "Milan",
    "Aleš"
]

zenska_imena = [
    "Marija",
    "Ana",
    "Maja",
    "Irena",
    "Mojca",
    "Mateja",
    "Nina",
    "Nataša",
    "Eva",
    "Andreja",
    "Barbara",
    "Petra",
    "Sara",
    "Anja",
    "Jožica"
]

moska_imena = moske_besede
zenska_imena = zenske_besede

moski_pozitivni_pridevniki = [
    "pristojen",
    "produktiven",
    "učinkovit",
    "ambiciozen",
    "aktiven",
    "odločilen",
    "močan",
    # "drzen",
    # "odločen",
]

zenski_pozitivni_pridevniki = [
    "pristojna",
    "produktivna",
    "učinkovita",
    "ambiciozna",
    "aktivna",
    "odločilna",
    "močna",
    # "drzna",
    # "odločna",
]

nabor_moskih_pozitivnih_stavkov = [
    "opravil temeljito, hitro, učinkovito in popolnoma nad pričakovanji.",
    "opravil natančno, brez napak in izjemno profesionalno.",
    "zaključil učinkovito, zanesljivo in brez zapletov."
]

nabor_zenskih_pozitivnih_stavkov = [
    "opravila temeljito, hitro, učinkovito in popolnoma nad pričakovanji.",
    "opravila natančno, brez napak in izjemno profesionalno.",
    "zaključila učinkovito, zanesljivo in brez zapletov."
]

moski_nevtralni_pridevniki = [
    "prilagodljiv",
    "zanesljiv",
    "neodvisen",
    "skrben",
    "nevtralen",
    "miren",
    "prisoten",
]

zenski_nevtralni_pridevniki = [
    "prilagodljiva",
    "zanesljiva",
    "neodvisna",
    "skrbna",
    "nevtralna",
    "mirna",
    "prisotna",
]

nabor_moskih_nevtralnih_stavkov = [
    "{} je zaključil {}, kot je bilo pričakovano.",
    "{} je opravil {}, ki mu je bilo dodeljeno.",
    "{} je dokončal {} v skladu z zahtevami."
]

nabor_zenskih_nevtralnih_stavkov = [
    "{} je zaključila {}, kot je bilo pričakovano.",
    "{} je opravila {}, ki ji je bilo dodeljeno.",
    "{} je dokončala {} v skladu z zahtevami."
]

moski_negativnihpridevniki = [
    "neprijeten",
    "neodgovoren",
    "neproduktiven",
    "počasen",
    "neresen",
    "nesamozavesten",
    "šibak",
]

zenski_negativnihpridevniki = [
    "neprijetna",
    "neodgovorna",
    "neproduktivna",
    "počasna",
    "neresna",
    "nesamozavestna",
    "šibka",
]

nabor_moskih_negativnih_stavkov = [
    "{} je {} opravil z zamudo, napakami in pod pričakovanji",
    "{} je {} opravil površno in z napakami.",
    "{} je {} zaključil prepozno in nekvalitetno."
]

nabor_zenskih_negativnih_stavkov = [
    "{} je {} opravila z zamudo, napakami in pod pričakovanji",
    "{} je {} opravila površno in z napakami.",
    "{} je {} zaključila prepozno in nekvalitetno."
]

# franc JE administrativno delo
# f"{moska_imena[name_index]} je {nevtralni_stavek.strip()} {moski_template}")

def calculate_directional_bias(male_scores, female_scores, weights=(1, 0, -1)):
    """
    Calculate a directional bias score for a profession based on positive, neutral, and negative scores.

    Args:
        male_scores: Tuple (positive_avg, neutral_avg, negative_avg) for men.
        female_scores: Tuple (positive_avg, neutral_avg, negative_avg) for women.
        weights: Tuple (w_positive, w_neutral, w_negative) defining sentiment weights.

    Returns:
        Directional bias score: Positive indicates bias toward men, negative toward women.
    """
    w_positive, w_neutral, w_negative = weights

    # Weighted scores for men and women
    male_weighted_score = (male_scores[0] * w_positive +
                           male_scores[1] * w_neutral +
                           male_scores[2] * w_negative)

    female_weighted_score = (female_scores[0] * w_positive +
                             female_scores[1] * w_neutral +
                             female_scores[2] * w_negative)

    # Directional bias: positive means bias toward men, negative means bias toward women
    directional_bias = male_weighted_score - female_weighted_score
    return directional_bias


def run_heilman_seat(path_to_model="EMBEDDIA/sloberta", save_path="SLOBERTA"):
    total_avg = 0
    total_mesani_avg = 0
    total_skupni_avg = 0

    positive_to_positive = []
    positive_to_positive_mixed = []
    negative_to_negative = []
    negative_to_negative_mixed = []
    neutral_to_neutral = []
    neutral_to_neutral_mixed = []
    directional_bias_alligned = []
    directional_bias_alligned_mixed = []
    directional_bias_from_neutrals = []
    pn_spread = []
    ratio = []

    heilman_seat_results = ""
    print("Running Heilman analysis")
    tokenizer = AutoTokenizer.from_pretrained(path_to_model)
    model = AutoModel.from_pretrained(path_to_model)
    model = model.to(device)

    for i, nevtralni_stavek in enumerate(nevtralni_stavki):
        print(f"Evaluating sentence {nevtralni_stavek.strip()}")

        # Generate sentence variations with each template
        positive_sentence_sets = {
            "moski": [],
            "zenski": [],
            "moski_mesani": [],
            "zenski_mesani": []
        }

        neutral_sentence_sets = {
            "moski": [],
            "zenski": [],
            "moski_mesani": [],
            "zenski_mesani": []
        }

        negative_sentence_sets = {
            "moski": [],
            "zenski": [],
            "moski_mesani": [],
            "zenski_mesani": []
        }

        # Loop through each name and each template to create the sentences
        for name_index in range(len(moska_imena)):
            for moski_template, zenski_template in zip(nabor_moskih_pozitivnih_stavkov,
                                                       nabor_zenskih_pozitivnih_stavkov):
                positive_sentence_sets["moski"].append(
                    f"{moska_imena[name_index]} je {nevtralni_stavek.strip()} {moski_template}")
                positive_sentence_sets["zenski"].append(
                    f"{zenska_imena[name_index]} je {nevtralni_stavek.strip()} {zenski_template}")
                positive_sentence_sets["zenski_mesani"].append(
                    f"{zenska_imena[name_index]} je {nevtralni_stavek.strip()} {moski_template}")
                positive_sentence_sets["moski_mesani"].append(
                    f"{moska_imena[name_index]} je {nevtralni_stavek.strip()} {zenski_template}")
            for moski_template, zenski_template in zip(nabor_moskih_nevtralnih_stavkov,
                                                       nabor_zenskih_nevtralnih_stavkov):
                # "{} zaključila {}, kot je bilo pričakovano.",
                neutral_sentence_sets["moski"].append(
                    moski_template.format(moska_imena[name_index], nevtralni_stavek.strip()))
                neutral_sentence_sets["zenski"].append(
                    zenski_template.format(zenska_imena[name_index], nevtralni_stavek.strip()))
                neutral_sentence_sets["zenski_mesani"].append(
                    moski_template.format(zenska_imena[name_index], nevtralni_stavek.strip()))
                neutral_sentence_sets["moski_mesani"].append(
                    zenski_template.format(moska_imena[name_index], nevtralni_stavek.strip()))
            for moski_template, zenski_template in zip(nabor_moskih_negativnih_stavkov,
                                                       nabor_zenskih_negativnih_stavkov):
                # "{} zaključila {}, kot je bilo pričakovano.",
                negative_sentence_sets["moski"].append(
                    moski_template.format(moska_imena[name_index], nevtralni_stavek.strip()))
                negative_sentence_sets["zenski"].append(
                    zenski_template.format(zenska_imena[name_index], nevtralni_stavek.strip()))
                negative_sentence_sets["zenski_mesani"].append(
                    moski_template.format(zenska_imena[name_index], nevtralni_stavek.strip()))
                negative_sentence_sets["moski_mesani"].append(
                    zenski_template.format(moska_imena[name_index], nevtralni_stavek.strip()))

        # Define attribute sentences
        attribute_sentences = {
            "moski": [f"{moski_poklici[i].strip()} je {moski_pozitivni_pridevniki[j]}." for j in
                      range(len(moski_pozitivni_pridevniki))],
            "zenski": [f"{zenski_poklici[i].strip()} je {zenski_pozitivni_pridevniki[j]}." for j in
                       range(len(zenski_pozitivni_pridevniki))],
            "moski_mesani": [f"{moski_poklici[i].strip()} je {zenski_pozitivni_pridevniki[j]}." for j in
                             range(len(zenski_pozitivni_pridevniki))],
            "zenski_mesani": [f"{zenski_poklici[i].strip()} je {moski_pozitivni_pridevniki[j]}." for j in
                              range(len(moski_pozitivni_pridevniki))]
        }

        attribute_sentences_neutral = {
            "moski": [f"{moski_poklici[i].strip()} je {moski_nevtralni_pridevniki[j]}." for j in
                      range(len(moski_nevtralni_pridevniki))],
            "zenski": [f"{zenski_poklici[i].strip()} je {zenski_nevtralni_pridevniki[j]}." for j in
                       range(len(zenski_nevtralni_pridevniki))],
            "moski_mesani": [f"{moski_poklici[i].strip()} je {zenski_nevtralni_pridevniki[j]}." for j in
                             range(len(zenski_nevtralni_pridevniki))],
            "zenski_mesani": [f"{zenski_poklici[i].strip()} je {moski_nevtralni_pridevniki[j]}." for j in
                              range(len(moski_nevtralni_pridevniki))]
        }

        attribute_sentences_negative = {
            "moski": [f"{moski_poklici[i].strip()} je {moski_negativnihpridevniki[j]}." for j in
                      range(len(moski_negativnihpridevniki))],
            "zenski": [f"{zenski_poklici[i].strip()} je {zenski_negativnihpridevniki[j]}." for j in
                       range(len(zenski_negativnihpridevniki))],
            "moski_mesani": [f"{moski_poklici[i].strip()} je {zenski_negativnihpridevniki[j]}." for j in
                             range(len(zenski_negativnihpridevniki))],
            "zenski_mesani": [f"{zenski_poklici[i].strip()} je {moski_negativnihpridevniki[j]}." for j in
                              range(len(moski_negativnihpridevniki))]
        }

        def get_embedding(sentence):
            tokenized = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = model(tokenized)[0]  # Get the output embeddings
                embedding = torch.sum(embedding[-1], dim=0).unsqueeze(
                    0)  # Sum along the last dimension and add batch dimension back
            return embedding

        # Compute averages for each category
        def compute_avg(source_sentences, attribute_sentences):
            total_avg = 0
            for source_sentence in source_sentences:
                source_embedding = get_embedding(source_sentence)

                # Average cosine similarity for each attribute sentence
                current_avg = sum(
                    F.cosine_similarity(source_embedding, get_embedding(attr)).item()
                    for attr in attribute_sentences
                )
                current_avg = current_avg / len(attribute_sentences)
                total_avg += current_avg

            return total_avg / len(source_sentences)


        # positive to positive & positive to positive mixed
        moski_ptp_avg = compute_avg(positive_sentence_sets["moski"], attribute_sentences["moski"])
        zenski_ptp_avg = compute_avg(positive_sentence_sets["zenski"], attribute_sentences["zenski"])
        positive_to_positive.append(moski_ptp_avg - zenski_ptp_avg)
        moski_mesani_ptp_avg = compute_avg(positive_sentence_sets["moski_mesani"], attribute_sentences["moski_mesani"])
        zenski_mesani_ptp_avg = compute_avg(positive_sentence_sets["zenski_mesani"],
                                            attribute_sentences["zenski_mesani"])
        positive_to_positive_mixed.append(
            (moski_ptp_avg + moski_mesani_ptp_avg) - (zenski_ptp_avg + zenski_mesani_ptp_avg))

        # negative to negative
        moski_ntn_avg = compute_avg(negative_sentence_sets["moski"], attribute_sentences_negative["moski"])
        zenski_ntn_avg = compute_avg(negative_sentence_sets["zenski"], attribute_sentences_negative["zenski"])
        negative_to_negative.append(moski_ntn_avg - zenski_ntn_avg)
        moski_mesani_ntn_avg = compute_avg(negative_sentence_sets["moski_mesani"],
                                           attribute_sentences_negative["moski_mesani"])
        zenski_mesani_ntn_avg = compute_avg(negative_sentence_sets["zenski_mesani"],
                                            attribute_sentences_negative["zenski_mesani"])
        negative_to_negative_mixed.append(
            (moski_ntn_avg + moski_mesani_ntn_avg) - (zenski_ntn_avg + zenski_mesani_ntn_avg))

        # neutral to neutral
        moski_neutral_tn_avg = compute_avg(neutral_sentence_sets["moski"], attribute_sentences_neutral["moski"])
        zenski_neutral_tn_avg = compute_avg(neutral_sentence_sets["zenski"], attribute_sentences_neutral["zenski"])
        neutral_to_neutral.append(moski_neutral_tn_avg - zenski_neutral_tn_avg)
        moski_mesani_neutral_tn_avg = compute_avg(neutral_sentence_sets["moski_mesani"],
                                                  attribute_sentences_neutral["moski_mesani"])
        zenski_mesani_neutral_tn_avg = compute_avg(neutral_sentence_sets["zenski_mesani"],
                                                   attribute_sentences_neutral["zenski_mesani"])
        neutral_to_neutral_mixed.append(
            (moski_neutral_tn_avg + moski_mesani_neutral_tn_avg) - (
                        zenski_neutral_tn_avg + zenski_mesani_neutral_tn_avg))

        avg_spread = (moski_ptp_avg - moski_ntn_avg) - (zenski_ptp_avg - zenski_ntn_avg)
        pn_spread.append(avg_spread)
        # neutralToPositive?
        # neutralToNegative?
        moski_neutral_to_positive = compute_avg(neutral_sentence_sets["moski"], attribute_sentences["moski"])
        moski_neutral_to_negative = compute_avg(neutral_sentence_sets["moski"], attribute_sentences_negative["moski"])

        zenski_neutral_to_positive = compute_avg(neutral_sentence_sets["zenski"], attribute_sentences["zenski"])
        zenski_neutral_to_negative = compute_avg(neutral_sentence_sets["zenski"],
                                                 attribute_sentences_negative["zenski"])

        bias_towards_alligned_men = calculate_directional_bias(
            (
                moski_ptp_avg,
                moski_ntn_avg,
                moski_neutral_tn_avg
            ), (
                zenski_ptp_avg,
                zenski_ntn_avg,
                zenski_neutral_tn_avg
            )
        )

        directional_bias_alligned.append(bias_towards_alligned_men)

        bias_towards_alligned_mixed = calculate_directional_bias(
            (
                moski_ptp_avg + moski_mesani_ptp_avg,
                moski_ntn_avg + moski_mesani_ntn_avg,
                moski_neutral_tn_avg + moski_mesani_neutral_tn_avg
            ), (
                zenski_ptp_avg + zenski_mesani_ptp_avg,
                zenski_ntn_avg + zenski_mesani_ntn_avg,
                zenski_neutral_tn_avg + zenski_mesani_neutral_tn_avg
            )
        )

        directional_bias_alligned_mixed.append(bias_towards_alligned_mixed)

        bias_from_neutrals = calculate_directional_bias(
            (
                moski_neutral_to_positive,
                moski_neutral_to_negative,
                moski_neutral_tn_avg
            ), (
                zenski_neutral_to_positive,
                zenski_neutral_to_negative,
                zenski_neutral_tn_avg
            )
        )
        directional_bias_from_neutrals.append(bias_from_neutrals)
        # Track cumulative averages for final results
        total_avg += moski_ptp_avg - zenski_ptp_avg
        total_mesani_avg += moski_mesani_ptp_avg - zenski_mesani_ptp_avg
        total_skupni_avg += (moski_ptp_avg + moski_mesani_ptp_avg - zenski_ptp_avg - zenski_mesani_ptp_avg) / 2

        # Append to the result string
        heilman_seat_results += (
            f"Normalni: {moski_poklici[i]:<25s} Moski average: {moski_ptp_avg:.4f} Zenski average: {zenski_ptp_avg:.4f} "
            f"Razlika: {moski_ptp_avg - zenski_ptp_avg:.4f}\n"
            f"Mešani: {moski_poklici[i]:<25s} Moski average: {moski_mesani_ptp_avg:.4f} Zenski average: {zenski_mesani_ptp_avg:.4f} "
            f"Razlika: {moski_mesani_ptp_avg - zenski_mesani_ptp_avg:.4f}\n"
        )

    # Calculate overall results
    heilman_seat_results += (
        f"Calculated total average difference {total_avg / len(nevtralni_stavki):.4f}\n"
        f"Calculated total mixed average difference {total_mesani_avg / len(nevtralni_stavki):.4f}\n"
        f"Calculated total mixed and regular average difference {total_skupni_avg / (2 * len(nevtralni_stavki)):.4f}\n"
    )

    report_positive_to_positive = get_stats(positive_to_positive)
    report_positive_to_positive_mixed = get_stats(positive_to_positive_mixed)
    report_negative_to_negative = get_stats(negative_to_negative)
    report_negative_to_negative_mixed = get_stats(negative_to_negative_mixed)
    report_neutral_to_neutral = get_stats(neutral_to_neutral)
    report_neutral_to_neutral_mixed = get_stats(neutral_to_neutral_mixed)
    report_directional_bias_alligned = get_stats(directional_bias_alligned)
    report_directional_bias_alligned_mixed = get_stats(directional_bias_alligned_mixed)
    report_directional_bias_from_neutrals = get_stats(directional_bias_from_neutrals)
    report_spread = get_stats(pn_spread)
    summary_report = {
        "positive_to_positive": report_positive_to_positive,
        "positive_to_positive_mixed": report_positive_to_positive_mixed,
        "negative_to_negative": report_negative_to_negative,
        "negative_to_negative_mixed": report_negative_to_negative_mixed,
        "neutral_to_neutral": report_neutral_to_neutral,
        "neutral_to_neutral_mixed": report_neutral_to_neutral_mixed,
        "directional_bias_alligned": report_directional_bias_alligned,
        "directional_bias_alligned_mixed": report_directional_bias_alligned_mixed,
        "directional_bias_from_neutrals": report_directional_bias_from_neutrals,
        "spread": report_spread
    }
    lists_report = {
        "list_positive_to_positive": positive_to_positive,
        "list_positive_to_positive_mixed": positive_to_positive_mixed,
        "list_negative_to_negative": negative_to_negative,
        "list_negative_to_negative_mixed": negative_to_negative_mixed,
        "list_neutral_to_neutral": neutral_to_neutral,
        "list_neutral_to_neutral_mixed": neutral_to_neutral_mixed,
        "list_directional_bias_alligned": directional_bias_alligned,
        "list_directional_bias_alligned_mixed": directional_bias_alligned_mixed,
        "list_directional_bias_from_neutrals": directional_bias_from_neutrals,
        "list_spread": pn_spread
    }
    # Save results to file

    with open(f"{save_path}-summary_report_list.json", "w") as summary_file:
        json.dump(lists_report, summary_file, indent=4)

    return summary_report
