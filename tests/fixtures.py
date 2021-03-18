import os
from os.path import dirname, join, realpath

from server.models import ModelProperty
from server.models import datatypes as dt


def large_db(partitioned_db):
    import generator

    base_path = dirname(realpath(__file__))
    data_path = join(base_path, "data", "large")
    seed_file = join(data_path, "seed.json")
    generator.load(partitioned_db, seed_file, verbose=False, base_dir=data_path)
    return partitioned_db


def movie_db(partitioned_db):
    pdb = partitioned_db

    # [Model:IMDB Page]
    imdb_page = partitioned_db.create_model(
        "imdb_page", "IMDB page", "Page on http://www.imdb.com"
    )
    imdb_page_properties = {
        p.name: p
        for p in pdb.update_properties(
            imdb_page,
            ModelProperty(
                name="url",
                display_name="URL",
                data_type=dt.String(),
                description="URL",
                model_title=True,
            ),
            ModelProperty(
                name="published",
                display_name="Published",
                data_type=dt.Boolean(),
                description="Published",
            ),
        )
    }

    # [Model:Wiki Page]
    wiki_page = partitioned_db.create_model(
        "wiki_page", "Wikipedia page", "Page on http://www.wikipedia.org"
    )
    wiki_page_properties = {
        p.name: p
        for p in pdb.update_properties(
            wiki_page,
            ModelProperty(
                name="url",
                display_name="URL",
                data_type=dt.String(),
                description="URL",
                model_title=True,
            ),
            ModelProperty(
                name="published",
                display_name="Published",
                data_type=dt.Boolean(),
                description="Published",
            ),
        )
    }

    # [Model:Movie]
    movie = partitioned_db.create_model("movie", "Movie", "A Movie")
    movie_properties = {
        p.name: p
        for p in pdb.update_properties(
            movie,
            ModelProperty(
                name="title",
                display_name="Title",
                data_type=dt.String(),
                description="Title",
                model_title=True,
            ),
            ModelProperty(
                name="description",
                display_name="Description",
                data_type=dt.String(),
                required=False,
                default_value="--NO DESCRIPTION PROVIDED--",
            ),
            ModelProperty(
                name="released",
                display_name="Released",
                data_type=dt.Long(),
                description="Year of release",
            ),
            ModelProperty(
                name="date_of_release",
                display_name="Date of release",
                data_type=dt.Date(),
                description="Date of releasee",
            ),
            ModelProperty(
                name="tag_line",
                display_name="Tag Line",
                data_type=dt.String(),
                description="Tag line",
            ),
            ModelProperty(
                name="rating",
                display_name="Rating",
                data_type=dt.Enumeration(
                    items=dt.String(),
                    enum=["Unwatchable", "Poor", "Fair", "Good", "Excellent"],
                ),
                description="Rating",
            ),
            ModelProperty(
                name="tags",
                display_name="Tags",
                data_type=dt.Array(items=dt.String()),
                description="Tags",
                required=False,
            ),
        )
    }

    # [Model:Person]
    person = partitioned_db.create_model("person", "Person", "Person")
    person_properties = {
        p.name: p
        for p in pdb.update_properties(
            person,
            ModelProperty(
                name="name",
                display_name="Name",
                data_type=dt.String(),
                description="Name",
                model_title=True,
            ),
            ModelProperty(
                name="born",
                display_name="Born",
                data_type=dt.Long(),
                description="Born",
            ),
        )
    }

    # [Model:Genre]
    genre = partitioned_db.create_model("genre", "Genre", "Genre")
    genre_properties = {
        p.name: p
        for p in pdb.update_properties(
            genre,
            ModelProperty(
                name="name",
                display_name="Name",
                data_type=dt.String(),
                description="Name",
                model_title=True,
            ),
        )
    }

    acted_in = pdb.create_model_relationship(person, "acted_in", movie)
    has_leading_role = pdb.create_model_relationship(person, "has_leading_role", movie)
    directed = pdb.create_model_relationship(person, "directed", movie)
    reviewed = pdb.create_model_relationship(person, "reviewed", movie)  # noqa: F841
    produced = pdb.create_model_relationship(person, "produced", movie)
    wrote = pdb.create_model_relationship(person, "wrote", movie)
    categorized_as = pdb.create_model_relationship(movie, "categorized_as", genre)
    # "Linked properties":
    has_imdb_entry = pdb.create_model_relationship(
        person, "has_imdb_entry", imdb_page, one_to_many=False
    )
    has_wiki_entry = pdb.create_model_relationship(
        person, "has_wiki_entry", wiki_page, one_to_many=False
    )

    Action, Comedy, Drama, Horror, Romance, SciFi = pdb.create_records(
        genre,
        [
            {"name": "Action"},
            {"name": "Comedy"},
            {"name": "Drama"},
            {"name": "Horror"},
            {"name": "Romance"},
            {"name": "Science Fiction"},
        ],
    )

    award = partitioned_db.create_model("award", "Award", "Award")
    award_properties = {
        p.name: p
        for p in pdb.update_properties(
            award,
            ModelProperty(
                name="name",
                display_name="Name",
                data_type=dt.String(),
                description="Name",
                model_title=True,
            ),
        )
    }
    AcademyAward, BAFTAAward = pdb.create_records(
        award, [{"name": "Academy Award"}, {"name": "BAFTA Award"}]
    )
    person_won_award = pdb.create_model_relationship(person, "won", award)
    movie_won_award = pdb.create_model_relationship(movie, "won", award)

    # CREATE (TheMatrix:Movie {title:'The Matrix', released:1999, tagline:'Welcome to the Real World'})
    # CREATE (Keanu:Person {name:'Keanu Reeves', born:1964})
    # CREATE (Carrie:Person {name:'Carrie-Anne Moss', born:1967})
    # CREATE (Laurence:Person {name:'Laurence Fishburne', born:1961})
    # CREATE (Hugo:Person {name:'Hugo Weaving', born:1960})
    # CREATE (LillyW:Person {name:'Lilly Wachowski', born:1967})
    # CREATE (LanaW:Person {name:'Lana Wachowski', born:1965})
    # CREATE (JoelS:Person {name:'Joel Silver', born:1952})
    # CREATE
    #   (Keanu)-[:ACTED_IN {roles:['Neo']}]->(TheMatrix),
    #   (Carrie)-[:ACTED_IN {roles:['Trinity']}]->(TheMatrix),
    #   (Laurence)-[:ACTED_IN {roles:['Morpheus']}]->(TheMatrix),
    #   (Hugo)-[:ACTED_IN {roles:['Agent Smith']}]->(TheMatrix),
    #   (LillyW)-[:DIRECTED]->(TheMatrix),
    #   (LanaW)-[:DIRECTED]->(TheMatrix),
    #   (JoelS)-[:PRODUCED]->(TheMatrix)
    (TheMatrix,) = pdb.create_records(
        movie,
        [
            {
                "title": "The Matrix",
                "released": 1999,
                "date_of_release": "1999-03-31",
                "tag_line": "Welcome to the Real World",
                "rating": "Good",
                "tags": [
                    "dystopia",
                    "hacker",
                    "scifi",
                    "future",
                    "noir",
                    "blockbuster",
                    "spoon",
                ],
            }
        ],
    )
    Keanu, Carrie, Laurence, Hugo, LillyW, LanaW, JoelS = pdb.create_records(
        person,
        [
            {"name": "Keanu Reeves", "born": 1964},
            {"name": "Carrie-Anne Moss", "born": 1967},
            {"name": "Laurence Fishburne", "born": 1961},
            {"name": "Hugo Weaving", "born": 1960},
            {"name": "Lilly Wachowski", "born": 1967},
            {"name": "Lana Wachowski", "born": 1965},
            {"name": "Joel Silver", "born": 1952},
        ],
    )

    KeanuIMDB, LaurenceIMDB, HugoIMDB, LanaWIMDB, LillyWIMDB = pdb.create_records(
        imdb_page,
        [
            {"url": "https://www.imdb.com/name/nm0000206", "published": True},
            {"url": "https://www.imdb.com/name/nm0000401", "published": True},
            {"url": "https://www.imdb.com/name/nm0915989", "published": False},
            {"url": "https://www.imdb.com/name/nm0905154", "published": True},
            {"url": "https://www.imdb.com/name/nm0905152", "published": False},
        ],
    )
    KeanuWiki, LaurenceWiki, HugoWiki, LanaWWiki, LillyWWiki = pdb.create_records(
        wiki_page,
        [
            {"url": "https://en.wikipedia.org/wiki/Keanu_Reeves", "published": True},
            {
                "url": "https://en.wikipedia.org/wiki/Laurence_Fishburne",
                "published": True,
            },
            {"url": "https://en.wikipedia.org/wiki/Hugo_Weaving", "published": False},
            {"url": "https://en.wikipedia.org/wiki/The_Wachowskis", "published": True},
            {"url": "https://en.wikipedia.org/wiki/The_Wachowskis", "published": True},
        ],
    )

    pdb.create_record_relationship(Keanu, has_imdb_entry, KeanuIMDB)
    pdb.create_record_relationship(Laurence, has_imdb_entry, LaurenceIMDB)
    pdb.create_record_relationship(Hugo, has_imdb_entry, HugoIMDB)
    pdb.create_record_relationship(LanaW, has_imdb_entry, LanaWIMDB)
    pdb.create_record_relationship(LillyW, has_imdb_entry, LillyWIMDB)

    pdb.create_record_relationship(Keanu, has_wiki_entry, KeanuWiki)
    pdb.create_record_relationship(Laurence, has_wiki_entry, LaurenceWiki)
    pdb.create_record_relationship(Hugo, has_wiki_entry, HugoWiki)
    pdb.create_record_relationship(LanaW, has_wiki_entry, LanaWWiki)
    pdb.create_record_relationship(LillyW, has_wiki_entry, LillyWWiki)

    pdb.create_record_relationship(TheMatrix, categorized_as, Action)
    pdb.create_record_relationship(TheMatrix, categorized_as, SciFi)
    pdb.create_record_relationship(TheMatrix, movie_won_award, AcademyAward)
    pdb.create_record_relationship(TheMatrix, movie_won_award, BAFTAAward)
    pdb.create_record_relationship(Keanu, acted_in, TheMatrix)
    pdb.create_record_relationship(Carrie, acted_in, TheMatrix)
    pdb.create_record_relationship(Laurence, acted_in, TheMatrix)
    pdb.create_record_relationship(Hugo, acted_in, TheMatrix)
    pdb.create_record_relationship(LillyW, directed, TheMatrix)
    pdb.create_record_relationship(LanaW, directed, TheMatrix)
    pdb.create_record_relationship(JoelS, produced, TheMatrix)
    pdb.create_record_relationship(LillyW, person_won_award, AcademyAward)
    pdb.create_record_relationship(LanaW, person_won_award, AcademyAward)

    pdb.create_record_relationship(Keanu, has_leading_role, TheMatrix)

    # CREATE (Emil:Person {name:"Emil Eifrem", born:1978})
    # CREATE (Emil)-[:ACTED_IN {roles:["Emil"]}]->(TheMatrix)
    #
    # CREATE (TheMatrixReloaded:Movie {title:'The Matrix Reloaded', released:2003, tagline:'Free your mind'})
    # CREATE
    #   (Keanu)-[:ACTED_IN {roles:['Neo']}]->(TheMatrixReloaded),
    #   (Carrie)-[:ACTED_IN {roles:['Trinity']}]->(TheMatrixReloaded),
    #   (Laurence)-[:ACTED_IN {roles:['Morpheus']}]->(TheMatrixReloaded),
    #   (Hugo)-[:ACTED_IN {roles:['Agent Smith']}]->(TheMatrixReloaded),
    #   (LillyW)-[:DIRECTED]->(TheMatrixReloaded),
    #   (LanaW)-[:DIRECTED]->(TheMatrixReloaded),
    #   (JoelS)-[:PRODUCED]->(TheMatrixReloaded)

    (TheMatrixReloaded,) = pdb.create_records(
        movie,
        [
            {
                "title": "The Matrix Reloaded",
                "released": 2003,
                "date_of_release": "2003-05-15",
                "tag_line": "Free your mind",
                "rating": "Fair",
                "tags": ["dystopia", "oracle", "scifi", "future", "noir", "prophecy"],
            }
        ],
    )
    (Emil,) = pdb.create_records(person, [{"name": "Emil Eifrem", "born": 1978}])
    pdb.create_record_relationship(TheMatrixReloaded, categorized_as, Action)
    pdb.create_record_relationship(TheMatrixReloaded, categorized_as, SciFi)
    pdb.create_record_relationship(Emil, acted_in, TheMatrixReloaded)
    pdb.create_record_relationship(Keanu, acted_in, TheMatrixReloaded)
    pdb.create_record_relationship(Carrie, acted_in, TheMatrixReloaded)
    pdb.create_record_relationship(Laurence, acted_in, TheMatrixReloaded)
    pdb.create_record_relationship(Hugo, acted_in, TheMatrixReloaded)
    pdb.create_record_relationship(LillyW, directed, TheMatrixReloaded)
    pdb.create_record_relationship(LanaW, directed, TheMatrixReloaded)
    pdb.create_record_relationship(JoelS, produced, TheMatrixReloaded)

    # CREATE (TheMatrixRevolutions:Movie {title:'The Matrix Revolutions', released:2003, tagline:'Everything that has a beginning has an end'})
    # CREATE
    #   (Keanu)-[:ACTED_IN {roles:['Neo']}]->(TheMatrixRevolutions),
    #   (Carrie)-[:ACTED_IN {roles:['Trinity']}]->(TheMatrixRevolutions),
    #   (Laurence)-[:ACTED_IN {roles:['Morpheus']}]->(TheMatrixRevolutions),
    #   (Hugo)-[:ACTED_IN {roles:['Agent Smith']}]->(TheMatrixRevolutions),
    #   (LillyW)-[:DIRECTED]->(TheMatrixRevolutions),
    #   (LanaW)-[:DIRECTED]->(TheMatrixRevolutions),
    #   (JoelS)-[:PRODUCED]->(TheMatrixRevolutions)

    (TheMatrixRevolutions,) = pdb.create_records(
        movie,
        [
            {
                "title": "The Matrix Revolutions",
                "released": 2003,
                "date_of_release": "2003-11-05",
                "tag_line": "Everything that has a beginning has an end",
                "rating": "Unwatchable",
                "tags": ["dystopia", "machine", "scifi", "future", "fight"],
            }
        ],
    )
    pdb.create_record_relationship(TheMatrixRevolutions, categorized_as, Action)
    pdb.create_record_relationship(TheMatrixRevolutions, categorized_as, SciFi)
    pdb.create_record_relationship(Keanu, acted_in, TheMatrixRevolutions)
    pdb.create_record_relationship(Carrie, acted_in, TheMatrixRevolutions)
    pdb.create_record_relationship(Laurence, acted_in, TheMatrixRevolutions)
    pdb.create_record_relationship(Hugo, acted_in, TheMatrixRevolutions)
    pdb.create_record_relationship(LillyW, directed, TheMatrixRevolutions)
    pdb.create_record_relationship(LanaW, directed, TheMatrixRevolutions)
    pdb.create_record_relationship(JoelS, produced, TheMatrixRevolutions)

    # CREATE (TheDevilsAdvocate:Movie {title:"The Devil's Advocate", released:1997, tagline:'Evil has its winning ways'})
    # CREATE (Charlize:Person {name:'Charlize Theron', born:1975})
    # CREATE (Al:Person {name:'Al Pacino', born:1940})
    # CREATE (Taylor:Person {name:'Taylor Hackford', born:1944})
    # CREATE
    #   (Keanu)-[:ACTED_IN {roles:['Kevin Lomax']}]->(TheDevilsAdvocate),
    #   (Charlize)-[:ACTED_IN {roles:['Mary Ann Lomax']}]->(TheDevilsAdvocate),
    #   (Al)-[:ACTED_IN {roles:['John Milton']}]->(TheDevilsAdvocate),
    #   (Taylor)-[:DIRECTED]->(TheDevilsAdvocate)
    (TheDevilsAdvocate,) = pdb.create_records(
        movie,
        [
            {
                "title": "The Devil's Advocate",
                "released": 1997,
                "date_of_release": "1997-10-17",
                "tag_line": "Evil has its winning ways",
                "rating": "Good",
                "tags": ["lawyer", "devil", "evil", "vanity", "money", "deal"],
            }
        ],
    )
    Charlize, Al, Taylor = pdb.create_records(
        person,
        [
            {"name": "Charlize Theron", "born": 1975},
            {"name": "Al Pacino", "born": 1940},
            {"name": "Taylor Hackford", "born": 1944},
        ],
    )
    pdb.create_record_relationship(TheDevilsAdvocate, categorized_as, Drama)
    pdb.create_record_relationship(TheDevilsAdvocate, categorized_as, Horror)
    pdb.create_record_relationship(Keanu, acted_in, TheDevilsAdvocate)
    pdb.create_record_relationship(Charlize, acted_in, TheDevilsAdvocate)
    pdb.create_record_relationship(Al, acted_in, TheDevilsAdvocate)
    pdb.create_record_relationship(Taylor, directed, TheDevilsAdvocate)
    pdb.create_record_relationship(TheDevilsAdvocate, movie_won_award, BAFTAAward)

    # CREATE (AFewGoodMen:Movie {title:"A Few Good Men", released:1992, tagline:"In the heart of the nation's capital, in a courthouse of the U.S. government, one man will stop at nothing to keep his honor, and one will stop at nothing to find the truth."})
    # CREATE (TomC:Person {name:'Tom Cruise', born:1962})
    # CREATE (JackN:Person {name:'Jack Nicholson', born:1937})
    # CREATE (DemiM:Person {name:'Demi Moore', born:1962})
    # CREATE (KevinB:Person {name:'Kevin Bacon', born:1958})
    # CREATE (KieferS:Person {name:'Kiefer Sutherland', born:1966})
    # CREATE (NoahW:Person {name:'Noah Wyle', born:1971})
    # CREATE (CubaG:Person {name:'Cuba Gooding Jr.', born:1968})
    # CREATE (KevinP:Person {name:'Kevin Pollak', born:1957})
    # CREATE (JTW:Person {name:'J.T. Walsh', born:1943})
    # CREATE (JamesM:Person {name:'James Marshall', born:1967})
    # CREATE (ChristopherG:Person {name:'Christopher Guest', born:1948})
    # CREATE (RobR:Person {name:'Rob Reiner', born:1947})
    # CREATE (AaronS:Person {name:'Aaron Sorkin', born:1961})
    # CREATE
    #   (TomC)-[:ACTED_IN {roles:['Lt. Daniel Kaffee']}]->(AFewGoodMen),
    #   (JackN)-[:ACTED_IN {roles:['Col. Nathan R. Jessup']}]->(AFewGoodMen),
    #   (DemiM)-[:ACTED_IN {roles:['Lt. Cdr. JoAnne Galloway']}]->(AFewGoodMen),
    #   (KevinB)-[:ACTED_IN {roles:['Capt. Jack Ross']}]->(AFewGoodMen),
    #   (KieferS)-[:ACTED_IN {roles:['Lt. Jonathan Kendrick']}]->(AFewGoodMen),
    #   (NoahW)-[:ACTED_IN {roles:['Cpl. Jeffrey Barnes']}]->(AFewGoodMen),
    #   (CubaG)-[:ACTED_IN {roles:['Cpl. Carl Hammaker']}]->(AFewGoodMen),
    #   (KevinP)-[:ACTED_IN {roles:['Lt. Sam Weinberg']}]->(AFewGoodMen),
    #   (JTW)-[:ACTED_IN {roles:['Lt. Col. Matthew Andrew Markinson']}]->(AFewGoodMen),
    #   (JamesM)-[:ACTED_IN {roles:['Pfc. Louden Downey']}]->(AFewGoodMen),
    #   (ChristopherG)-[:ACTED_IN {roles:['Dr. Stone']}]->(AFewGoodMen),
    #   (AaronS)-[:ACTED_IN {roles:['Man in Bar']}]->(AFewGoodMen),
    #   (RobR)-[:DIRECTED]->(AFewGoodMen),
    #   (AaronS)-[:WROTE]->(AFewGoodMen)
    (AFewGoodMen,) = pdb.create_records(
        movie,
        [
            {
                "title": "A Few Good Men",
                "released": 1992,
                "date_of_release": "1992-12-09",
                "tag_line": "In the heart of the nation's capital, in a courthouse of the U.S. government, one man will stop at nothing to keep his honor, and one will stop at nothing to find the truth",
                "rating": "Good",
                "tags": ["lawyer", "judge", "honor", "justice", "duty", "military"],
            }
        ],
    )
    (
        TomC,
        JackN,
        DemiM,
        KevinB,
        KieferS,
        NoahW,
        CubaG,
        KevinP,
        JTW,
        JamesM,
        ChristopherG,
        RobR,
        AaronS,
    ) = pdb.create_records(
        person,
        [
            {"name": "Tom Cruise", "born": 1962},
            {"name": "Jack Nicholson", "born": 1937},
            {"name": "Demi Moore", "born": 1962},
            {"name": "Kevin Bacon", "born": 1958},
            {"name": "Kiefer Sutherland", "born": 1966},
            {"name": "Noah Wyle", "born": 1971},
            {"name": "Cuba Gooding Jr.", "born": 1968},
            {"name": "Kevin Pollak", "born": 1957},
            {"name": "J.T. Walsh", "born": 1943},
            {"name": "James Marshall", "born": 1967},
            {"name": "Christopher Guest", "born": 1948},
            {"name": "Rob Reiner", "born": 1947},
            {"name": "Aaron Sorkin", "born": 1961},
        ],
    )
    pdb.create_record_relationship(AFewGoodMen, categorized_as, Drama)
    pdb.create_record_relationship(TomC, acted_in, AFewGoodMen)
    pdb.create_record_relationship(JackN, acted_in, AFewGoodMen)
    pdb.create_record_relationship(DemiM, acted_in, AFewGoodMen)
    pdb.create_record_relationship(KevinB, acted_in, AFewGoodMen)
    pdb.create_record_relationship(KieferS, acted_in, AFewGoodMen)
    pdb.create_record_relationship(NoahW, acted_in, AFewGoodMen)
    pdb.create_record_relationship(CubaG, acted_in, AFewGoodMen)
    pdb.create_record_relationship(KevinP, acted_in, AFewGoodMen)
    pdb.create_record_relationship(JTW, acted_in, AFewGoodMen)
    pdb.create_record_relationship(JamesM, acted_in, AFewGoodMen)
    pdb.create_record_relationship(ChristopherG, acted_in, AFewGoodMen)
    pdb.create_record_relationship(AaronS, acted_in, AFewGoodMen)
    pdb.create_record_relationship(RobR, directed, AFewGoodMen)
    pdb.create_record_relationship(AaronS, wrote, AFewGoodMen)

    # CREATE (TopGun:Movie {title:"Top Gun", released:1986, tagline:'I feel the need, the need for speed.'})
    # CREATE (KellyM:Person {name:'Kelly McGillis', born:1957})
    # CREATE (ValK:Person {name:'Val Kilmer', born:1959})
    # CREATE (AnthonyE:Person {name:'Anthony Edwards', born:1962})
    # CREATE (TomS:Person {name:'Tom Skerritt', born:1933})
    # CREATE (MegR:Person {name:'Meg Ryan', born:1961})
    # CREATE (TonyS:Person {name:'Tony Scott', born:1944})
    # CREATE (JimC:Person {name:'Jim Cash', born:1941})
    # CREATE
    #   (TomC)-[:ACTED_IN {roles:['Maverick']}]->(TopGun),
    #   (KellyM)-[:ACTED_IN {roles:['Charlie']}]->(TopGun),
    #   (ValK)-[:ACTED_IN {roles:['Iceman']}]->(TopGun),
    #   (AnthonyE)-[:ACTED_IN {roles:['Goose']}]->(TopGun),
    #   (TomS)-[:ACTED_IN {roles:['Viper']}]->(TopGun),
    #   (MegR)-[:ACTED_IN {roles:['Carole']}]->(TopGun),
    #   (TonyS)-[:DIRECTED]->(TopGun),
    #   (JimC)-[:WROTE]->(TopGun)
    (TopGun,) = pdb.create_records(
        movie,
        [
            {
                "title": "Top Gun",
                "released": 1986,
                "date_of_release": "1986-05-16",
                "tag_line": "I feel the need, the need for speed.",
                "tags": ["pilot", "jet", "flying", "rivalry"],
            }
        ],
    )
    KellyM, ValK, AnthonyE, TomS, MegR, TonyS, JimC = pdb.create_records(
        person,
        [
            {"name": "Kelly McGillis", "born": 1956},
            {"name": "Val Kilmer", "born": 1959},
            {"name": "Anthony Edwards", "born": 1962},
            {"name": "Tom Skerritt", "born": 1933},
            {"name": "Meg Ryan", "born": 1961},
            {"name": "Tony Scott", "born": 1944},
            {"name": "Jim Cash", "born": 1941},
        ],
    )
    pdb.create_record_relationship(TopGun, categorized_as, Action)
    pdb.create_record_relationship(TomC, acted_in, TopGun)
    pdb.create_record_relationship(KellyM, acted_in, TopGun)
    pdb.create_record_relationship(ValK, acted_in, TopGun)
    pdb.create_record_relationship(AnthonyE, acted_in, TopGun)
    pdb.create_record_relationship(TomS, acted_in, TopGun)
    pdb.create_record_relationship(MegR, acted_in, TopGun)
    pdb.create_record_relationship(TonyS, directed, TopGun)
    pdb.create_record_relationship(JimC, wrote, TopGun)

    # CREATE (JerryMaguire:Movie {title:'Jerry Maguire', released:2000, tagline:'The rest of his life begins now.'})
    # CREATE (ReneeZ:Person {name:'Renee Zellweger', born:1969})
    # CREATE (KellyP:Person {name:'Kelly Preston', born:1962})
    # CREATE (JerryO:Person {name:"Jerry O'Connell", born:1974})
    # CREATE (JayM:Person {name:'Jay Mohr', born:1970})
    # CREATE (BonnieH:Person {name:'Bonnie Hunt', born:1961})
    # CREATE (ReginaK:Person {name:'Regina King', born:1971})
    # CREATE (JonathanL:Person {name:'Jonathan Lipnicki', born:1996})
    # CREATE (CameronC:Person {name:'Cameron Crowe', born:1957})
    # CREATE
    #   (TomC)-[:ACTED_IN {roles:['Jerry Maguire']}]->(JerryMaguire),
    #   (CubaG)-[:ACTED_IN {roles:['Rod Tidwell']}]->(JerryMaguire),
    #   (ReneeZ)-[:ACTED_IN {roles:['Dorothy Boyd']}]->(JerryMaguire),
    #   (KellyP)-[:ACTED_IN {roles:['Avery Bishop']}]->(JerryMaguire),
    #   (JerryO)-[:ACTED_IN {roles:['Frank Cushman']}]->(JerryMaguire),
    #   (JayM)-[:ACTED_IN {roles:['Bob Sugar']}]->(JerryMaguire),
    #   (BonnieH)-[:ACTED_IN {roles:['Laurel Boyd']}]->(JerryMaguire),
    #   (ReginaK)-[:ACTED_IN {roles:['Marcee Tidwell']}]->(JerryMaguire),
    #   (JonathanL)-[:ACTED_IN {roles:['Ray Boyd']}]->(JerryMaguire),
    #   (CameronC)-[:DIRECTED]->(JerryMaguire),
    #   (CameronC)-[:PRODUCED]->(JerryMaguire),
    #   (CameronC)-[:WROTE]->(JerryMaguire)
    (JerryMaguire,) = pdb.create_records(
        movie,
        [
            {
                "title": "Jerry Maguire",
                "released": 2000,
                "date_of_release": "1996-12-16",
                "tag_line": "The rest of his life begins now.",
                "rating": "Good",
                "tags": ["sports", "agent", "football"],
            }
        ],
    )
    (
        ReneeZ,
        KellyP,
        JerryO,
        JayM,
        BonnieH,
        ReginaK,
        JonathanL,
        CameronC,
    ) = pdb.create_records(
        person,
        [
            {"name": "Renee Zellweger", "born": 1969},
            {"name": "Kelly Preston", "born": 1962},
            {"name": "Jerry O'Connell", "born": 1974},
            {"name": "Jay Mohr", "born": 1970},
            {"name": "Bonnie Hunt", "born": 1961},
            {"name": "Regina King", "born": 1971},
            {"name": "Jonathan Lipnicki", "born": 1996},
            {"name": "Cameron Crowe", "born": 1957},
        ],
    )
    pdb.create_record_relationship(JerryMaguire, categorized_as, Comedy)
    pdb.create_record_relationship(TomC, acted_in, JerryMaguire)
    pdb.create_record_relationship(CubaG, acted_in, JerryMaguire)
    pdb.create_record_relationship(ReneeZ, acted_in, JerryMaguire)
    pdb.create_record_relationship(KellyP, acted_in, JerryMaguire)
    pdb.create_record_relationship(JerryO, acted_in, JerryMaguire)
    pdb.create_record_relationship(JayM, acted_in, JerryMaguire)
    pdb.create_record_relationship(BonnieH, acted_in, JerryMaguire)
    pdb.create_record_relationship(ReginaK, acted_in, JerryMaguire)
    pdb.create_record_relationship(JonathanL, acted_in, JerryMaguire)
    pdb.create_record_relationship(CameronC, directed, JerryMaguire)
    pdb.create_record_relationship(CameronC, produced, JerryMaguire)
    pdb.create_record_relationship(CameronC, wrote, JerryMaguire)

    # CREATE (StandByMe:Movie {title:"Stand By Me", released:1986, tagline:"For some, it's the last real taste of innocence, and the first real taste of life. But for everyone, it's the time that memories are made of."})
    # CREATE (RiverP:Person {name:'River Phoenix', born:1970})
    # CREATE (CoreyF:Person {name:'Corey Feldman', born:1971})
    # CREATE (WilW:Person {name:'Wil Wheaton', born:1972})
    # CREATE (JohnC:Person {name:'John Cusack', born:1966})
    # CREATE (MarshallB:Person {name:'Marshall Bell', born:1942})
    # CREATE
    #   (WilW)-[:ACTED_IN {roles:['Gordie Lachance']}]->(StandByMe),
    #   (RiverP)-[:ACTED_IN {roles:['Chris Chambers']}]->(StandByMe),
    #   (JerryO)-[:ACTED_IN {roles:['Vern Tessio']}]->(StandByMe),
    #   (CoreyF)-[:ACTED_IN {roles:['Teddy Duchamp']}]->(StandByMe),
    #   (JohnC)-[:ACTED_IN {roles:['Denny Lachance']}]->(StandByMe),
    #   (KieferS)-[:ACTED_IN {roles:['Ace Merrill']}]->(StandByMe),
    #   (MarshallB)-[:ACTED_IN {roles:['Mr. Lachance']}]->(StandByMe),
    #   (RobR)-[:DIRECTED]->(StandByMe)
    (StandByMe,) = pdb.create_records(
        movie,
        [
            {
                "title": "Stand By Me",
                "released": 1986,
                "date_of_release": "1986-08-22",
                "tag_line": "For some, it's the last real taste of innocence, and the first real taste of life. But for everyone, it's the time that memories are made of.",
                "rating": "Excellent",
                "tags": ["friends", "childhood", "woods", "summer", "innocence"],
            }
        ],
    )
    RiverP, CoreyF, WilW, JohnC, MarshallB = pdb.create_records(
        person,
        [
            {"name": "River Phoenix", "born": 1969},
            {"name": "Corey Feldman", "born": 1962},
            {"name": "Wil Wheaton", "born": 1972},
            {"name": "John Cusack", "born": 1966},
            {"name": "Marshall Bell", "born": 1942},
        ],
    )
    pdb.create_record_relationship(StandByMe, categorized_as, Drama)
    pdb.create_record_relationship(RiverP, acted_in, StandByMe)
    pdb.create_record_relationship(CoreyF, acted_in, StandByMe)
    pdb.create_record_relationship(WilW, acted_in, StandByMe)
    pdb.create_record_relationship(JohnC, acted_in, StandByMe)
    pdb.create_record_relationship(KieferS, acted_in, StandByMe)
    pdb.create_record_relationship(MarshallB, acted_in, StandByMe)
    pdb.create_record_relationship(RobR, directed, StandByMe)

    # CREATE (AsGoodAsItGets:Movie {title:'As Good as It Gets', released:1997, tagline:'A comedy from the heart that goes for the throat.'})
    # CREATE (HelenH:Person {name:'Helen Hunt', born:1963})
    # CREATE (GregK:Person {name:'Greg Kinnear', born:1963})
    # CREATE (JamesB:Person {name:'James L. Brooks', born:1940})
    # CREATE
    #   (JackN)-[:ACTED_IN {roles:['Melvin Udall']}]->(AsGoodAsItGets),
    #   (HelenH)-[:ACTED_IN {roles:['Carol Connelly']}]->(AsGoodAsItGets),
    #   (GregK)-[:ACTED_IN {roles:['Simon Bishop']}]->(AsGoodAsItGets),
    #   (CubaG)-[:ACTED_IN {roles:['Frank Sachs']}]->(AsGoodAsItGets),
    #   (JamesB)-[:DIRECTED]->(AsGoodAsItGets)
    (AsGoodAsItGets,) = pdb.create_records(
        movie,
        [
            {
                "title": "As Good as It Gets",
                "released": 1997,
                "date_of_release": "1997-12-06",
                "tag_line": "A comedy from the heart that goes for the throat.",
                "rating": "Fair",
                "tags": ["writer", "dog", "author", "new york"],
            }
        ],
    )
    HelenH, GregK, JamesB = pdb.create_records(
        person,
        [
            {"name": "Helen Hunt", "born": 1963},
            {"name": "Greg Kinnear", "born": 1963},
            {"name": "James L. Brooks", "born": 1940},
        ],
    )
    pdb.create_record_relationship(AsGoodAsItGets, categorized_as, Comedy)
    pdb.create_record_relationship(JackN, acted_in, AsGoodAsItGets)
    pdb.create_record_relationship(HelenH, acted_in, AsGoodAsItGets)
    pdb.create_record_relationship(GregK, acted_in, AsGoodAsItGets)
    pdb.create_record_relationship(CubaG, acted_in, AsGoodAsItGets)
    pdb.create_record_relationship(JamesB, directed, AsGoodAsItGets)

    # CREATE (WhatDreamsMayCome:Movie {title:'What Dreams May Come', released:1998, tagline:'After life there is more. The end is just the beginning.'})
    # CREATE (AnnabellaS:Person {name:'Annabella Sciorra', born:1960})
    # CREATE (MaxS:Person {name:'Max von Sydow', born:1929})
    # CREATE (WernerH:Person {name:'Werner Herzog', born:1942})
    # CREATE (Robin:Person {name:'Robin Williams', born:1951})
    # CREATE (VincentW:Person {name:'Vincent Ward', born:1956})
    # CREATE
    #   (Robin)-[:ACTED_IN {roles:['Chris Nielsen']}]->(WhatDreamsMayCome),
    #   (CubaG)-[:ACTED_IN {roles:['Albert Lewis']}]->(WhatDreamsMayCome),
    #   (AnnabellaS)-[:ACTED_IN {roles:['Annie Collins-Nielsen']}]->(WhatDreamsMayCome),
    #   (MaxS)-[:ACTED_IN {roles:['The Tracker']}]->(WhatDreamsMayCome),
    #   (WernerH)-[:ACTED_IN {roles:['The Face']}]->(WhatDreamsMayCome),
    #   (VincentW)-[:DIRECTED]->(WhatDreamsMayCome)
    (WhatDreamsMayCome,) = pdb.create_records(
        movie,
        [
            {
                "title": "What Dreams May Come",
                "released": 1998,
                "date_of_release": "1998-10-02",
                "tag_line": "After life there is more. The end is just the beginning.",
                "rating": "Poor",
                "tags": ["heaven", "death", "afterlife", "hell", "mortality"],
            }
        ],
    )
    AnnabellaS, MaxS, WernerH, Robin, VincentW = pdb.create_records(
        person,
        [
            {"name": "Annabella Sciorra", "born": 1960},
            {"name": "Max von Sydow", "born": 1929},
            {"name": "Werner Herzog", "born": 1942},
            {"name": "Robin Williams", "born": 1951},
            {"name": "Vincent Ward", "born": 1956},
        ],
    )
    pdb.create_record_relationship(WhatDreamsMayCome, categorized_as, Drama)
    pdb.create_record_relationship(Robin, acted_in, WhatDreamsMayCome)
    pdb.create_record_relationship(CubaG, acted_in, WhatDreamsMayCome)
    pdb.create_record_relationship(AnnabellaS, acted_in, WhatDreamsMayCome)
    pdb.create_record_relationship(MaxS, acted_in, WhatDreamsMayCome)
    pdb.create_record_relationship(WernerH, directed, WhatDreamsMayCome)
    pdb.create_record_relationship(VincentW, directed, WhatDreamsMayCome)

    # CREATE (SnowFallingonCedars:Movie {title:'Snow Falling on Cedars', released:1999, tagline:'First loves last. Forever.'})
    # CREATE (EthanH:Person {name:'Ethan Hawke', born:1970})
    # CREATE (RickY:Person {name:'Rick Yune', born:1971})
    # CREATE (JamesC:Person {name:'James Cromwell', born:1940})
    # CREATE (ScottH:Person {name:'Scott Hicks', born:1953})
    # CREATE
    #   (EthanH)-[:ACTED_IN {roles:['Ishmael Chambers']}]->(SnowFallingonCedars),
    #   (RickY)-[:ACTED_IN {roles:['Kazuo Miyamoto']}]->(SnowFallingonCedars),
    #   (MaxS)-[:ACTED_IN {roles:['Nels Gudmundsson']}]->(SnowFallingonCedars),
    #   (JamesC)-[:ACTED_IN {roles:['Judge Fielding']}]->(SnowFallingonCedars),
    #   (ScottH)-[:DIRECTED]->(SnowFallingonCedars)
    (SnowFallingonCedars,) = pdb.create_records(
        movie,
        [
            {
                "title": "Snow Falling on Cedars",
                "released": 1999,
                "date_of_release": "1999-09-01",
                "tag_line": "First loves last. Forever.",
                "rating": "Fair",
                "tags": [
                    "injustice",
                    "fisherman",
                    "murder",
                    "japanese",
                    "island",
                    "1950s",
                ],
            }
        ],
    )
    EthanH, RickY, JamesC, ScottH = pdb.create_records(
        person,
        [
            {"name": "Ethan Hawke", "born": 1970},
            {"name": "Rick Yune", "born": 1971},
            {"name": "James Cromwel", "born": 1940},
            {"name": "Scott Hicks", "born": 1953},
        ],
    )
    pdb.create_record_relationship(SnowFallingonCedars, categorized_as, Drama)
    pdb.create_record_relationship(EthanH, acted_in, SnowFallingonCedars)
    pdb.create_record_relationship(RickY, acted_in, SnowFallingonCedars)
    pdb.create_record_relationship(MaxS, acted_in, SnowFallingonCedars)
    pdb.create_record_relationship(JamesC, acted_in, SnowFallingonCedars)
    pdb.create_record_relationship(ScottH, directed, SnowFallingonCedars)

    # CREATE (YouveGotMail:Movie {title:"You've Got Mail", released:1998, tagline:'At odds in life... in love on-line.'})
    # CREATE (ParkerP:Person {name:'Parker Posey', born:1968})
    # CREATE (DaveC:Person {name:'Dave Chappelle', born:1973})
    # CREATE (SteveZ:Person {name:'Steve Zahn', born:1967})
    # CREATE (TomH:Person {name:'Tom Hanks', born:1956})
    # CREATE (NoraE:Person {name:'Nora Ephron', born:1941})
    # CREATE
    #   (TomH)-[:ACTED_IN {roles:['Joe Fox']}]->(YouveGotMail),
    #   (MegR)-[:ACTED_IN {roles:['Kathleen Kelly']}]->(YouveGotMail),
    #   (GregK)-[:ACTED_IN {roles:['Frank Navasky']}]->(YouveGotMail),
    #   (ParkerP)-[:ACTED_IN {roles:['Patricia Eden']}]->(YouveGotMail),
    #   (DaveC)-[:ACTED_IN {roles:['Kevin Jackson']}]->(YouveGotMail),
    #   (SteveZ)-[:ACTED_IN {roles:['George Pappas']}]->(YouveGotMail),
    #   (NoraE)-[:DIRECTED]->(YouveGotMail)
    (YouveGotMail,) = pdb.create_records(
        movie,
        [
            {
                "title": "You've Got Mail",
                "released": 1998,
                "date_of_release": "1998-12-18",
                "tag_line": "At odds in life... in love on-line.",
                "rating": "Good",
                "tags": ["dog", "internet", "email", "bookstore", "new york", "email"],
            }
        ],
    )
    ParkerP, DaveC, SteveZ, TomH, NoraE = pdb.create_records(
        person,
        [
            {"name": "Parker Posey", "born": 1968},
            {"name": "Dave Chappelle", "born": 1973},
            {"name": "Steve Zahn", "born": 1967},
            {"name": "Tom Hanks", "born": 1956},
            {"name": "Nora Ephron", "born": 1941},
        ],
    )
    pdb.create_record_relationship(YouveGotMail, categorized_as, Romance)
    pdb.create_record_relationship(YouveGotMail, categorized_as, Comedy)
    pdb.create_record_relationship(TomH, acted_in, YouveGotMail)
    pdb.create_record_relationship(MegR, acted_in, YouveGotMail)
    pdb.create_record_relationship(GregK, acted_in, YouveGotMail)
    pdb.create_record_relationship(ParkerP, acted_in, YouveGotMail)
    pdb.create_record_relationship(DaveC, acted_in, YouveGotMail)
    pdb.create_record_relationship(SteveZ, acted_in, YouveGotMail)
    pdb.create_record_relationship(NoraE, directed, YouveGotMail)

    # CREATE (SleeplessInSeattle:Movie {title:'Sleepless in Seattle', released:1993, tagline:'What if someone you never met, someone you never saw, someone you never knew was the only someone for you?'})
    # CREATE (RitaW:Person {name:'Rita Wilson', born:1956})
    # CREATE (BillPull:Person {name:'Bill Pullman', born:1953})
    # CREATE (VictorG:Person {name:'Victor Garber', born:1949})
    # CREATE (RosieO:Person {name:"Rosie O'Donnell", born:1962})
    # CREATE
    #   (TomH)-[:ACTED_IN {roles:['Sam Baldwin']}]->(SleeplessInSeattle),
    #   (MegR)-[:ACTED_IN {roles:['Annie Reed']}]->(SleeplessInSeattle),
    #   (RitaW)-[:ACTED_IN {roles:['Suzy']}]->(SleeplessInSeattle),
    #   (BillPull)-[:ACTED_IN {roles:['Walter']}]->(SleeplessInSeattle),
    #   (VictorG)-[:ACTED_IN {roles:['Greg']}]->(SleeplessInSeattle),
    #   (RosieO)-[:ACTED_IN {roles:['Becky']}]->(SleeplessInSeattle),
    #   (NoraE)-[:DIRECTED]->(SleeplessInSeattle)
    (SleeplessInSeattle,) = pdb.create_records(
        movie,
        [
            {
                "title": "Sleepless in Seattle",
                "released": 1993,
                "date_of_release": "1993-07-25",
                "tag_line": "What if someone you never met, someone you never saw, someone you never knew was the only someone for you?",
                "rating": "Good",
                "tags": ["seattle", "radioshow"],
            }
        ],
    )
    RitaW, BillPull, VictorG, RosieO = pdb.create_records(
        person,
        [
            {"name": "Rita Wilson", "born": 1956},
            {"name": "Bill Pullman", "born": 1953},
            {"name": "Victor Garber", "born": 1949},
            {"name": "Rosie O'Donnell", "born": 1962},
        ],
    )
    pdb.create_record_relationship(SleeplessInSeattle, categorized_as, Romance)
    pdb.create_record_relationship(SleeplessInSeattle, categorized_as, Comedy)
    pdb.create_record_relationship(TomH, acted_in, SleeplessInSeattle)
    pdb.create_record_relationship(MegR, acted_in, SleeplessInSeattle)
    pdb.create_record_relationship(RitaW, acted_in, SleeplessInSeattle)
    pdb.create_record_relationship(BillPull, acted_in, SleeplessInSeattle)
    pdb.create_record_relationship(VictorG, acted_in, SleeplessInSeattle)
    pdb.create_record_relationship(RosieO, acted_in, SleeplessInSeattle)
    pdb.create_record_relationship(NoraE, directed, SleeplessInSeattle)

    # CREATE (JoeVersustheVolcano:Movie {title:'Joe Versus the Volcano', released:1990, tagline:'A story of love, lava and burning desire.'})
    # CREATE (JohnS:Person {name:'John Patrick Stanley', born:1950})
    # CREATE (Nathan:Person {name:'Nathan Lane', born:1956})
    # CREATE
    #   (TomH)-[:ACTED_IN {roles:['Joe Banks']}]->(JoeVersustheVolcano),
    #   (MegR)-[:ACTED_IN {roles:['DeDe', 'Angelica Graynamore', 'Patricia Graynamore']}]->(JoeVersustheVolcano),
    #   (Nathan)-[:ACTED_IN {roles:['Baw']}]->(JoeVersustheVolcano),
    #   (JohnS)-[:DIRECTED]->(JoeVersustheVolcano)
    #
    # CREATE (WhenHarryMetSally:Movie {title:'When Harry Met Sally', released:1998, tagline:'Can two friends sleep together and still love each other in the morning?'})
    # CREATE (BillyC:Person {name:'Billy Crystal', born:1948})
    # CREATE (CarrieF:Person {name:'Carrie Fisher', born:1956})
    # CREATE (BrunoK:Person {name:'Bruno Kirby', born:1949})
    # CREATE
    #   (BillyC)-[:ACTED_IN {roles:['Harry Burns']}]->(WhenHarryMetSally),
    #   (MegR)-[:ACTED_IN {roles:['Sally Albright']}]->(WhenHarryMetSally),
    #   (CarrieF)-[:ACTED_IN {roles:['Marie']}]->(WhenHarryMetSally),
    #   (BrunoK)-[:ACTED_IN {roles:['Jess']}]->(WhenHarryMetSally),
    #   (RobR)-[:DIRECTED]->(WhenHarryMetSally),
    #   (RobR)-[:PRODUCED]->(WhenHarryMetSally),
    #   (NoraE)-[:PRODUCED]->(WhenHarryMetSally),
    #   (NoraE)-[:WROTE]->(WhenHarryMetSally)
    #
    # CREATE (ThatThingYouDo:Movie {title:'That Thing You Do', released:1996, tagline:'In every life there comes a time when that thing you dream becomes that thing you do'})
    # CREATE (LivT:Person {name:'Liv Tyler', born:1977})
    # CREATE
    #   (TomH)-[:ACTED_IN {roles:['Mr. White']}]->(ThatThingYouDo),
    #   (LivT)-[:ACTED_IN {roles:['Faye Dolan']}]->(ThatThingYouDo),
    #   (Charlize)-[:ACTED_IN {roles:['Tina']}]->(ThatThingYouDo),
    #   (TomH)-[:DIRECTED]->(ThatThingYouDo)
    #
    # CREATE (TheReplacements:Movie {title:'The Replacements', released:2000, tagline:'Pain heals, Chicks dig scars... Glory lasts forever'})
    # CREATE (Brooke:Person {name:'Brooke Langton', born:1970})
    # CREATE (Gene:Person {name:'Gene Hackman', born:1930})
    # CREATE (Orlando:Person {name:'Orlando Jones', born:1968})
    # CREATE (Howard:Person {name:'Howard Deutch', born:1950})
    # CREATE
    #   (Keanu)-[:ACTED_IN {roles:['Shane Falco']}]->(TheReplacements),
    #   (Brooke)-[:ACTED_IN {roles:['Annabelle Farrell']}]->(TheReplacements),
    #   (Gene)-[:ACTED_IN {roles:['Jimmy McGinty']}]->(TheReplacements),
    #   (Orlando)-[:ACTED_IN {roles:['Clifford Franklin']}]->(TheReplacements),
    #   (Howard)-[:DIRECTED]->(TheReplacements)
    #
    # CREATE (RescueDawn:Movie {title:'RescueDawn', released:2006, tagline:"Based on the extraordinary true story of one man's fight for freedom"})
    # CREATE (ChristianB:Person {name:'Christian Bale', born:1974})
    # CREATE (ZachG:Person {name:'Zach Grenier', born:1954})
    # CREATE
    #   (MarshallB)-[:ACTED_IN {roles:['Admiral']}]->(RescueDawn),
    #   (ChristianB)-[:ACTED_IN {roles:['Dieter Dengler']}]->(RescueDawn),
    #   (ZachG)-[:ACTED_IN {roles:['Squad Leader']}]->(RescueDawn),
    #   (SteveZ)-[:ACTED_IN {roles:['Duane']}]->(RescueDawn),
    #   (WernerH)-[:DIRECTED]->(RescueDawn)
    #
    # CREATE (TheBirdcage:Movie {title:'The Birdcage', released:1996, tagline:'Come as you are'})
    # CREATE (MikeN:Person {name:'Mike Nichols', born:1931})
    # CREATE
    #   (Robin)-[:ACTED_IN {roles:['Armand Goldman']}]->(TheBirdcage),
    #   (Nathan)-[:ACTED_IN {roles:['Albert Goldman']}]->(TheBirdcage),
    #   (Gene)-[:ACTED_IN {roles:['Sen. Kevin Keeley']}]->(TheBirdcage),
    #   (MikeN)-[:DIRECTED]->(TheBirdcage)
    #
    # CREATE (Unforgiven:Movie {title:'Unforgiven', released:1992, tagline:"It's a hell of a thing, killing a man"})
    # CREATE (RichardH:Person {name:'Richard Harris', born:1930})
    # CREATE (ClintE:Person {name:'Clint Eastwood', born:1930})
    # CREATE
    #   (RichardH)-[:ACTED_IN {roles:['English Bob']}]->(Unforgiven),
    #   (ClintE)-[:ACTED_IN {roles:['Bill Munny']}]->(Unforgiven),
    #   (Gene)-[:ACTED_IN {roles:['Little Bill Daggett']}]->(Unforgiven),
    #   (ClintE)-[:DIRECTED]->(Unforgiven)
    #
    # CREATE (JohnnyMnemonic:Movie {title:'Johnny Mnemonic', released:1995, tagline:'The hottest data on earth. In the coolest head in town'})
    # CREATE (Takeshi:Person {name:'Takeshi Kitano', born:1947})
    # CREATE (Dina:Person {name:'Dina Meyer', born:1968})
    # CREATE (IceT:Person {name:'Ice-T', born:1958})
    # CREATE (RobertL:Person {name:'Robert Longo', born:1953})
    # CREATE
    #   (Keanu)-[:ACTED_IN {roles:['Johnny Mnemonic']}]->(JohnnyMnemonic),
    #   (Takeshi)-[:ACTED_IN {roles:['Takahashi']}]->(JohnnyMnemonic),
    #   (Dina)-[:ACTED_IN {roles:['Jane']}]->(JohnnyMnemonic),
    #   (IceT)-[:ACTED_IN {roles:['J-Bone']}]->(JohnnyMnemonic),
    #   (RobertL)-[:DIRECTED]->(JohnnyMnemonic)
    #
    # CREATE (CloudAtlas:Movie {title:'Cloud Atlas', released:2012, tagline:'Everything is connected'})
    # CREATE (HalleB:Person {name:'Halle Berry', born:1966})
    # CREATE (JimB:Person {name:'Jim Broadbent', born:1949})
    # CREATE (TomT:Person {name:'Tom Tykwer', born:1965})
    # CREATE (DavidMitchell:Person {name:'David Mitchell', born:1969})
    # CREATE (StefanArndt:Person {name:'Stefan Arndt', born:1961})
    # CREATE
    #   (TomH)-[:ACTED_IN {roles:['Zachry', 'Dr. Henry Goose', 'Isaac Sachs', 'Dermot Hoggins']}]->(CloudAtlas),
    #   (Hugo)-[:ACTED_IN {roles:['Bill Smoke', 'Haskell Moore', 'Tadeusz Kesselring', 'Nurse Noakes', 'Boardman Mephi', 'Old Georgie']}]->(CloudAtlas),
    #   (HalleB)-[:ACTED_IN {roles:['Luisa Rey', 'Jocasta Ayrs', 'Ovid', 'Meronym']}]->(CloudAtlas),
    #   (JimB)-[:ACTED_IN {roles:['Vyvyan Ayrs', 'Captain Molyneux', 'Timothy Cavendish']}]->(CloudAtlas),
    #   (TomT)-[:DIRECTED]->(CloudAtlas),
    #   (LillyW)-[:DIRECTED]->(CloudAtlas),
    #   (LanaW)-[:DIRECTED]->(CloudAtlas),
    #   (DavidMitchell)-[:WROTE]->(CloudAtlas),
    #   (StefanArndt)-[:PRODUCED]->(CloudAtlas)
    #
    # CREATE (TheDaVinciCode:Movie {title:'The Da Vinci Code', released:2006, tagline:'Break The Codes'})
    # CREATE (IanM:Person {name:'Ian McKellen', born:1939})
    # CREATE (AudreyT:Person {name:'Audrey Tautou', born:1976})
    # CREATE (PaulB:Person {name:'Paul Bettany', born:1971})
    # CREATE (RonH:Person {name:'Ron Howard', born:1954})
    # CREATE
    #   (TomH)-[:ACTED_IN {roles:['Dr. Robert Langdon']}]->(TheDaVinciCode),
    #   (IanM)-[:ACTED_IN {roles:['Sir Leight Teabing']}]->(TheDaVinciCode),
    #   (AudreyT)-[:ACTED_IN {roles:['Sophie Neveu']}]->(TheDaVinciCode),
    #   (PaulB)-[:ACTED_IN {roles:['Silas']}]->(TheDaVinciCode),
    #   (RonH)-[:DIRECTED]->(TheDaVinciCode)
    #
    # CREATE (VforVendetta:Movie {title:'V for Vendetta', released:2006, tagline:'Freedom! Forever!'})
    # CREATE (NatalieP:Person {name:'Natalie Portman', born:1981})
    # CREATE (StephenR:Person {name:'Stephen Rea', born:1946})
    # CREATE (JohnH:Person {name:'John Hurt', born:1940})
    # CREATE (BenM:Person {name: 'Ben Miles', born:1967})
    # CREATE
    #   (Hugo)-[:ACTED_IN {roles:['V']}]->(VforVendetta),
    #   (NatalieP)-[:ACTED_IN {roles:['Evey Hammond']}]->(VforVendetta),
    #   (StephenR)-[:ACTED_IN {roles:['Eric Finch']}]->(VforVendetta),
    #   (JohnH)-[:ACTED_IN {roles:['High Chancellor Adam Sutler']}]->(VforVendetta),
    #   (BenM)-[:ACTED_IN {roles:['Dascomb']}]->(VforVendetta),
    #   (JamesM)-[:DIRECTED]->(VforVendetta),
    #   (LillyW)-[:PRODUCED]->(VforVendetta),
    #   (LanaW)-[:PRODUCED]->(VforVendetta),
    #   (JoelS)-[:PRODUCED]->(VforVendetta),
    #   (LillyW)-[:WROTE]->(VforVendetta),
    #   (LanaW)-[:WROTE]->(VforVendetta)
    #
    # CREATE (SpeedRacer:Movie {title:'Speed Racer', released:2008, tagline:'Speed has no limits'})
    # CREATE (EmileH:Person {name:'Emile Hirsch', born:1985})
    # CREATE (JohnG:Person {name:'John Goodman', born:1960})
    # CREATE (SusanS:Person {name:'Susan Sarandon', born:1946})
    # CREATE (MatthewF:Person {name:'Matthew Fox', born:1966})
    # CREATE (ChristinaR:Person {name:'Christina Ricci', born:1980})
    # CREATE (Rain:Person {name:'Rain', born:1982})
    # CREATE
    #   (EmileH)-[:ACTED_IN {roles:['Speed Racer']}]->(SpeedRacer),
    #   (JohnG)-[:ACTED_IN {roles:['Pops']}]->(SpeedRacer),
    #   (SusanS)-[:ACTED_IN {roles:['Mom']}]->(SpeedRacer),
    #   (MatthewF)-[:ACTED_IN {roles:['Racer X']}]->(SpeedRacer),
    #   (ChristinaR)-[:ACTED_IN {roles:['Trixie']}]->(SpeedRacer),
    #   (Rain)-[:ACTED_IN {roles:['Taejo Togokahn']}]->(SpeedRacer),
    #   (BenM)-[:ACTED_IN {roles:['Cass Jones']}]->(SpeedRacer),
    #   (LillyW)-[:DIRECTED]->(SpeedRacer),
    #   (LanaW)-[:DIRECTED]->(SpeedRacer),
    #   (LillyW)-[:WROTE]->(SpeedRacer),
    #   (LanaW)-[:WROTE]->(SpeedRacer),
    #   (JoelS)-[:PRODUCED]->(SpeedRacer)
    #
    # CREATE (NinjaAssassin:Movie {title:'Ninja Assassin', released:2009, tagline:'Prepare to enter a secret world of assassins'})
    # CREATE (NaomieH:Person {name:'Naomie Harris'})
    # CREATE
    #   (Rain)-[:ACTED_IN {roles:['Raizo']}]->(NinjaAssassin),
    #   (NaomieH)-[:ACTED_IN {roles:['Mika Coretti']}]->(NinjaAssassin),
    #   (RickY)-[:ACTED_IN {roles:['Takeshi']}]->(NinjaAssassin),
    #   (BenM)-[:ACTED_IN {roles:['Ryan Maslow']}]->(NinjaAssassin),
    #   (JamesM)-[:DIRECTED]->(NinjaAssassin),
    #   (LillyW)-[:PRODUCED]->(NinjaAssassin),
    #   (LanaW)-[:PRODUCED]->(NinjaAssassin),
    #   (JoelS)-[:PRODUCED]->(NinjaAssassin)
    #
    # CREATE (TheGreenMile:Movie {title:'The Green Mile', released:1999, tagline:"Walk a mile you'll never forget."})
    # CREATE (MichaelD:Person {name:'Michael Clarke Duncan', born:1957})
    # CREATE (DavidM:Person {name:'David Morse', born:1953})
    # CREATE (SamR:Person {name:'Sam Rockwell', born:1968})
    # CREATE (GaryS:Person {name:'Gary Sinise', born:1955})
    # CREATE (PatriciaC:Person {name:'Patricia Clarkson', born:1959})
    # CREATE (FrankD:Person {name:'Frank Darabont', born:1959})
    # CREATE
    #   (TomH)-[:ACTED_IN {roles:['Paul Edgecomb']}]->(TheGreenMile),
    #   (MichaelD)-[:ACTED_IN {roles:['John Coffey']}]->(TheGreenMile),
    #   (DavidM)-[:ACTED_IN {roles:['Brutus "Brutal" Howell']}]->(TheGreenMile),
    #   (BonnieH)-[:ACTED_IN {roles:['Jan Edgecomb']}]->(TheGreenMile),
    #   (JamesC)-[:ACTED_IN {roles:['Warden Hal Moores']}]->(TheGreenMile),
    #   (SamR)-[:ACTED_IN {roles:['"Wild Bill" Wharton']}]->(TheGreenMile),
    #   (GaryS)-[:ACTED_IN {roles:['Burt Hammersmith']}]->(TheGreenMile),
    #   (PatriciaC)-[:ACTED_IN {roles:['Melinda Moores']}]->(TheGreenMile),
    #   (FrankD)-[:DIRECTED]->(TheGreenMile)
    #
    # CREATE (FrostNixon:Movie {title:'Frost/Nixon', released:2008, tagline:'400 million people were waiting for the truth.'})
    # CREATE (FrankL:Person {name:'Frank Langella', born:1938})
    # CREATE (MichaelS:Person {name:'Michael Sheen', born:1969})
    # CREATE (OliverP:Person {name:'Oliver Platt', born:1960})
    # CREATE
    #   (FrankL)-[:ACTED_IN {roles:['Richard Nixon']}]->(FrostNixon),
    #   (MichaelS)-[:ACTED_IN {roles:['David Frost']}]->(FrostNixon),
    #   (KevinB)-[:ACTED_IN {roles:['Jack Brennan']}]->(FrostNixon),
    #   (OliverP)-[:ACTED_IN {roles:['Bob Zelnick']}]->(FrostNixon),
    #   (SamR)-[:ACTED_IN {roles:['James Reston, Jr.']}]->(FrostNixon),
    #   (RonH)-[:DIRECTED]->(FrostNixon)
    #
    # CREATE (Hoffa:Movie {title:'Hoffa', released:1992, tagline:"He didn't want law. He wanted justice."})
    # CREATE (DannyD:Person {name:'Danny DeVito', born:1944})
    # CREATE (JohnR:Person {name:'John C. Reilly', born:1965})
    # CREATE
    #   (JackN)-[:ACTED_IN {roles:['Hoffa']}]->(Hoffa),
    #   (DannyD)-[:ACTED_IN {roles:['Robert "Bobby" Ciaro']}]->(Hoffa),
    #   (JTW)-[:ACTED_IN {roles:['Frank Fitzsimmons']}]->(Hoffa),
    #   (JohnR)-[:ACTED_IN {roles:['Peter "Pete" Connelly']}]->(Hoffa),
    #   (DannyD)-[:DIRECTED]->(Hoffa)
    #
    # CREATE (Apollo13:Movie {title:'Apollo 13', released:1995, tagline:'Houston, we have a problem.'})
    # CREATE (EdH:Person {name:'Ed Harris', born:1950})
    # CREATE (BillPax:Person {name:'Bill Paxton', born:1955})
    # CREATE
    #   (TomH)-[:ACTED_IN {roles:['Jim Lovell']}]->(Apollo13),
    #   (KevinB)-[:ACTED_IN {roles:['Jack Swigert']}]->(Apollo13),
    #   (EdH)-[:ACTED_IN {roles:['Gene Kranz']}]->(Apollo13),
    #   (BillPax)-[:ACTED_IN {roles:['Fred Haise']}]->(Apollo13),
    #   (GaryS)-[:ACTED_IN {roles:['Ken Mattingly']}]->(Apollo13),
    #   (RonH)-[:DIRECTED]->(Apollo13)
    #
    # CREATE (Twister:Movie {title:'Twister', released:1996, tagline:"Don't Breathe. Don't Look Back."})
    # CREATE (PhilipH:Person {name:'Philip Seymour Hoffman', born:1967})
    # CREATE (JanB:Person {name:'Jan de Bont', born:1943})
    # CREATE
    #   (BillPax)-[:ACTED_IN {roles:['Bill Harding']}]->(Twister),
    #   (HelenH)-[:ACTED_IN {roles:['Dr. Jo Harding']}]->(Twister),
    #   (ZachG)-[:ACTED_IN {roles:['Eddie']}]->(Twister),
    #   (PhilipH)-[:ACTED_IN {roles:['Dustin "Dusty" Davis']}]->(Twister),
    #   (JanB)-[:DIRECTED]->(Twister)
    #
    # CREATE (CastAway:Movie {title:'Cast Away', released:2000, tagline:'At the edge of the world, his journey begins.'})
    # CREATE (RobertZ:Person {name:'Robert Zemeckis', born:1951})
    # CREATE
    #   (TomH)-[:ACTED_IN {roles:['Chuck Noland']}]->(CastAway),
    #   (HelenH)-[:ACTED_IN {roles:['Kelly Frears']}]->(CastAway),
    #   (RobertZ)-[:DIRECTED]->(CastAway)
    #
    # CREATE (OneFlewOvertheCuckoosNest:Movie {title:"One Flew Over the Cuckoo's Nest", released:1975, tagline:"If he's crazy, what does that make you?"})
    # CREATE (MilosF:Person {name:'Milos Forman', born:1932})
    # CREATE
    #   (JackN)-[:ACTED_IN {roles:['Randle McMurphy']}]->(OneFlewOvertheCuckoosNest),
    #   (DannyD)-[:ACTED_IN {roles:['Martini']}]->(OneFlewOvertheCuckoosNest),
    #   (MilosF)-[:DIRECTED]->(OneFlewOvertheCuckoosNest)
    #
    # CREATE (SomethingsGottaGive:Movie {title:"Something's Gotta Give", released:2003})
    # CREATE (DianeK:Person {name:'Diane Keaton', born:1946})
    # CREATE (NancyM:Person {name:'Nancy Meyers', born:1949})
    # CREATE
    #   (JackN)-[:ACTED_IN {roles:['Harry Sanborn']}]->(SomethingsGottaGive),
    #   (DianeK)-[:ACTED_IN {roles:['Erica Barry']}]->(SomethingsGottaGive),
    #   (Keanu)-[:ACTED_IN {roles:['Julian Mercer']}]->(SomethingsGottaGive),
    #   (NancyM)-[:DIRECTED]->(SomethingsGottaGive),
    #   (NancyM)-[:PRODUCED]->(SomethingsGottaGive),
    #   (NancyM)-[:WROTE]->(SomethingsGottaGive)
    #
    # CREATE (BicentennialMan:Movie {title:'Bicentennial Man', released:1999, tagline:"One robot's 200 year journey to become an ordinary man."})
    # CREATE (ChrisC:Person {name:'Chris Columbus', born:1958})
    # CREATE
    #   (Robin)-[:ACTED_IN {roles:['Andrew Marin']}]->(BicentennialMan),
    #   (OliverP)-[:ACTED_IN {roles:['Rupert Burns']}]->(BicentennialMan),
    #   (ChrisC)-[:DIRECTED]->(BicentennialMan)
    #
    # CREATE (CharlieWilsonsWar:Movie {title:"Charlie Wilson's War", released:2007, tagline:"A stiff drink. A little mascara. A lot of nerve. Who said they couldn't bring down the Soviet empire."})
    # CREATE (JuliaR:Person {name:'Julia Roberts', born:1967})
    # CREATE
    #   (TomH)-[:ACTED_IN {roles:['Rep. Charlie Wilson']}]->(CharlieWilsonsWar),
    #   (JuliaR)-[:ACTED_IN {roles:['Joanne Herring']}]->(CharlieWilsonsWar),
    #   (PhilipH)-[:ACTED_IN {roles:['Gust Avrakotos']}]->(CharlieWilsonsWar),
    #   (MikeN)-[:DIRECTED]->(CharlieWilsonsWar)
    #
    # CREATE (ThePolarExpress:Movie {title:'The Polar Express', released:2004, tagline:'This Holiday Season… Believe'})
    # CREATE
    #   (TomH)-[:ACTED_IN {roles:['Hero Boy', 'Father', 'Conductor', 'Hobo', 'Scrooge', 'Santa Claus']}]->(ThePolarExpress),
    #   (RobertZ)-[:DIRECTED]->(ThePolarExpress)
    #
    # CREATE (ALeagueofTheirOwn:Movie {title:'A League of Their Own', released:1992, tagline:'Once in a lifetime you get a chance to do something different.'})
    # CREATE (Madonna:Person {name:'Madonna', born:1954})
    # CREATE (GeenaD:Person {name:'Geena Davis', born:1956})
    # CREATE (LoriP:Person {name:'Lori Petty', born:1963})
    # CREATE (PennyM:Person {name:'Penny Marshall', born:1943})
    # CREATE
    #   (TomH)-[:ACTED_IN {roles:['Jimmy Dugan']}]->(ALeagueofTheirOwn),
    #   (GeenaD)-[:ACTED_IN {roles:['Dottie Hinson']}]->(ALeagueofTheirOwn),
    #   (LoriP)-[:ACTED_IN {roles:['Kit Keller']}]->(ALeagueofTheirOwn),
    #   (RosieO)-[:ACTED_IN {roles:['Doris Murphy']}]->(ALeagueofTheirOwn),
    #   (Madonna)-[:ACTED_IN {roles:['"All the Way" Mae Mordabito']}]->(ALeagueofTheirOwn),
    #   (BillPax)-[:ACTED_IN {roles:['Bob Hinson']}]->(ALeagueofTheirOwn),
    #   (PennyM)-[:DIRECTED]->(ALeagueofTheirOwn)
    #
    # CREATE (PaulBlythe:Person {name:'Paul Blythe'})
    # CREATE (AngelaScope:Person {name:'Angela Scope'})
    # CREATE (JessicaThompson:Person {name:'Jessica Thompson'})
    # CREATE (JamesThompson:Person {name:'James Thompson'})
    #
    # CREATE
    #   (JamesThompson)-[:FOLLOWS]->(JessicaThompson),
    #   (AngelaScope)-[:FOLLOWS]->(JessicaThompson),
    #   (PaulBlythe)-[:FOLLOWS]->(AngelaScope)
    #
    # CREATE
    #   (JessicaThompson)-[:REVIEWED {summary:'An amazing journey', rating:95}]->(CloudAtlas),
    #   (JessicaThompson)-[:REVIEWED {summary:'Silly, but fun', rating:65}]->(TheReplacements),
    #   (JamesThompson)-[:REVIEWED {summary:'The coolest football movie ever', rating:100}]->(TheReplacements),
    #   (AngelaScope)-[:REVIEWED {summary:'Pretty funny at times', rating:62}]->(TheReplacements),
    #   (JessicaThompson)-[:REVIEWED {summary:'Dark, but compelling', rating:85}]->(Unforgiven),
    #   (JessicaThompson)-[:REVIEWED {summary:"Slapstick redeemed only by the Robin Williams and Gene Hackman's stellar performances", rating:45}]->(TheBirdcage),
    #   (JessicaThompson)-[:REVIEWED {summary:'A solid romp', rating:68}]->(TheDaVinciCode),
    #   (JamesThompson)-[:REVIEWED {summary:'Fun, but a little far fetched', rating:65}]->(TheDaVinciCode),
    #   (JessicaThompson)-[:REVIEWED {summary:'You had me at Jerry', rating:92}]->(JerryMaguire)
    #
    # WITH TomH as a
    # MATCH (a)-[:ACTED_IN]->(m)<-[:DIRECTED]-(d) RETURN a,m,d LIMIT 10
    # ;
    variables = locals()
    db = variables.pop("partitioned_db")
    variables.pop("pdb")

    return db, variables


def sample_patient_db(partitioned_db):
    patient = partitioned_db.create_model("patient", "Patient", "a person")
    partitioned_db.update_properties(
        patient,
        ModelProperty(
            name="name",
            display_name="Name",
            data_type=dt.String(),
            description="",
            model_title=True,
        ),
        ModelProperty(name="age", display_name="Age", data_type=dt.Long()),
    )

    visit = partitioned_db.create_model("visit", "Visit", "a visit")
    partitioned_db.update_properties(
        visit,
        ModelProperty(
            name="day",
            display_name="Day",
            data_type=dt.String(),
            description="",
            model_title=True,
        ),
    )

    medication = partitioned_db.create_model("medication", "Medication", "a medication")
    partitioned_db.update_properties(
        medication,
        ModelProperty(
            name="name",
            display_name="Name",
            data_type=dt.String(),
            description="",
            model_title=True,
        ),
    )

    attends = partitioned_db.create_model_relationship(patient, "attends", visit)
    prescribed = partitioned_db.create_model_relationship(
        visit, "prescribed", medication
    )

    alice, bob = partitioned_db.create_records(
        patient, [{"name": "Alice", "age": 34}, {"name": "Bob", "age": 20}]
    )
    monday, tuesday = partitioned_db.create_records(
        visit, [{"day": "Monday"}, {"day": "Tuesday"}]
    )
    aspirin, tylenol, motrin = partitioned_db.create_records(
        medication, [{"name": "Aspirin"}, {"name": "Tylenol"}, {"name": "Motrin"}]
    )

    partitioned_db.create_record_relationship(alice, attends, monday)
    partitioned_db.create_record_relationship(monday, prescribed, aspirin)

    partitioned_db.create_record_relationship(bob, attends, tuesday)
    partitioned_db.create_record_relationship(tuesday, prescribed, aspirin)
    partitioned_db.create_record_relationship(tuesday, prescribed, tylenol)

    return {
        "models": {"patient": patient, "medication": medication, "visit": visit},
        "relationships": {"attends": attends, "prescribed": prescribed},
        "records": {
            "alice": alice,
            "bob": bob,
            "monday": monday,
            "tuesday": tuesday,
            "aspirin": aspirin,
            "tylenol": tylenol,
            "motrin": motrin,
        },
    }
