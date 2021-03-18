import argparse
import json
import logging
import re
from csv import DictWriter
from dataclasses import InitVar, dataclass
from itertools import islice
from os.path import dirname, join, realpath
from random import choice, randint, shuffle
from tempfile import NamedTemporaryFile
from typing import Any, Callable, Dict, Generator, List, Optional, Union

from dataclasses_json import dataclass_json  # type: ignore
from mimesis import Generic
from more_itertools import unique_everseen

log = logging.getLogger(__file__)
log.setLevel(logging.INFO)


MAX_MODELS: int = 50
MAX_RECORDS: int = 1000


g = Generic("en")
GraphValue = Union[str, int]


@dataclass_json
@dataclass
class ModelProperty:
    """
    A generated model property
    """

    name: str
    display_name: str
    data_type: str
    description: str
    gen_value: InitVar[Callable[[], GraphValue]]

    def __post_init__(self, gen_value):
        self.generate = gen_value


def gen_property(
    name: Callable[[], GraphValue], datatype: str, gen_value: Any
) -> ModelProperty:
    def prop(g: Generic) -> ModelProperty:
        prop = name(g) if callable(name) else name
        return ModelProperty(
            name=prop.lower().replace(" ", "_"),
            display_name=prop.upper() if len(prop) <= 4 else prop.capitalize(),
            data_type=datatype,
            description=prop.capitalize(),
            gen_value=lambda: gen_value(g),
        )

    return prop


def model_name() -> str:
    """
    Generates a model name
    """
    return "_".join([g.text.word() for _ in range(randint(1, 3))])


__GENERATORS__: List[ModelProperty] = [
    gen_property("age", "Long", lambda g: g.person.age()),
    gen_property("atomic_number", "Long", lambda g: g.science.atomic_number()),
    gen_property("chemical_element", "String", lambda g: g.science.chemical_element()),
    gen_property("color", "String", lambda g: g.text.color()),
    gen_property("dna_sequence", "String", lambda g: g.science.dna_sequence()),
    gen_property("file_name", "String", lambda g: g.file.file_name()),
    gen_property("home", "String", lambda g: g.path.home()),
    gen_property("isbn", "String", lambda g: g.code.isbn()),
    gen_property("level", "String", lambda g: g.text.level()),
    gen_property(g.text.word(), "String", lambda g: g.cryptographic.mnemonic_phrase()),
    gen_property("mime_type", "String", lambda g: g.file.mime_type()),
    gen_property(
        "programming_language", "String", lambda g: g.development.programming_language()
    ),
    gen_property("rna_sequence", "String", lambda g: g.science.rna_sequence()),
    gen_property(
        "software_license", "String", lambda g: g.development.software_license()
    ),
    gen_property("university", "String", lambda g: g.person.university()),
    gen_property("user", "String", lambda g: g.path.user()),
    gen_property("vin", "String", lambda g: g.transport.vehicle_registration_code()),
    gen_property(lambda g: g.person.occupation(), "String", lambda g: g.person.name()),
] + [
    gen_property(g.text.word(), "String", lambda g: g.cryptographic.mnemonic_phrase())
    for _ in range(5)
]


def make_properties():
    # g = Generic("en", seed=randint(-0xFFFF, 0xFFFF))
    shuffle(__GENERATORS__)
    while True:
        yield choice(__GENERATORS__)(g)


@dataclass_json
@dataclass
class ModelRelationship:
    """
    A generated model property
    """

    type_: str
    to: str
    one_to_many: bool
    display_name: str


@dataclass_json
@dataclass
class Model:
    """
    A generated model
    """

    name: str
    display_name: str
    description: str
    id: Optional[str] = None

    def __post_init__(self):
        self.properties = list(
            unique_everseen(
                islice(make_properties(), randint(3, 16)), key=lambda p: p.name
            )
        )

    def connect(self, to) -> ModelRelationship:
        if to is None:
            raise Exception("Model.connect => to: other model's ID must be defined")

        type_ = (
            choice(["is", "has", "one_of", "not_a", "larger_than", "smaller_than"])
            + "_"
            + g.text.word()
        )
        type_ = type_.upper()

        return ModelRelationship(
            type=type,
            to=to,
            one_to_many=choice([False, True, True, True]),
            display_name=type.lower().capitalize(),
        )

    @property
    def records(self) -> Generator[Dict[str, GraphValue], None, None]:
        """
        Return a generator of records
        """
        while True:
            # A dict ensures all name keys are unique:
            entries = {p.name: p.generate() for p in self.properties}
            yield [dict(name=k, value=v) for (k, v) in entries.items()]


def create_model() -> Model:
    """
    Create a new model payload:

        {"name": "certificates", "displayName": "Certificates", "description": "..."}
    """
    name = model_name()
    return Model(
        name=name,
        display_name=re.sub(r"[-_]", " ", name.capitalize()),
        description=g.text.sentence(),
    )


def seed_data(model: Model, randomize: bool, max_records: int, verbose: bool = False):
    if randomize:
        record_count = randint(int(max_records * 0.66), max_records)
    else:
        record_count = max_records

    if verbose:
        print(f'Generating {record_count} records for model "{model.name}"')

    # Models
    with NamedTemporaryFile(delete=False, mode="w") as model_file:
        m = model.to_dict()
        model_writer = DictWriter(model_file, fieldnames=m.keys(), delimiter="|")
        model_writer.writeheader()
        model_writer.writerow(m)

    # Model properties
    with NamedTemporaryFile(delete=False, mode="w") as properties_file:
        properties_writer = DictWriter(
            properties_file,
            fieldnames=model.properties[0].to_dict().keys(),
            delimiter="|",
        )
        wrote_header = False
        for p in model.properties:
            p = p.to_dict()
            if not wrote_header:
                properties_writer.writeheader()
                wrote_header = True
            properties_writer.writerow(p)

    # Records
    with NamedTemporaryFile(delete=False, mode="w") as records_file:
        record_writer = DictWriter(records_file, fieldnames=["values"], delimiter="|")
        wrote_header = False
        for r in islice(model.records, record_count):
            if not wrote_header:
                record_writer.writeheader()
                wrote_header = True
            record_writer.writerow(dict(values=json.dumps(r)))

    return {
        "model": model_file.name,
        "properties": properties_file.name,
        "records": records_file.name,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate seed data")
    parser.add_argument("-m", "--max-models", type=int, default=MAX_MODELS)
    parser.add_argument("-r", "--max-records", type=int, default=MAX_RECORDS)
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "-o",
        "--out",
        type=str,
        default=realpath(join(dirname(__file__), "output", "seed.json")),
    )

    args = parser.parse_args()

    if args.randomize:
        model_count = randint(int(args.max_models * 0.66), args.max_models)
    else:
        model_count = args.max_models

    dest = args.out

    print(f"Generating {model_count} model(s)")
    print(f"Writing output to {dest}")

    with open(dest, "w") as output:
        json.dump(
            [
                seed_data(
                    create_model(),
                    randomize=args.randomize,
                    max_records=args.max_records,
                    verbose=args.verbose,
                )
                for _ in range(model_count)
            ],
            output,
        )
