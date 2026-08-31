"""Harmonize case taxonomy and add sourced preferred-foot metadata.

Every registered injury event is reviewed explicitly by player name. Preferred
foot is checked against EA SPORTS FC for the complete cohort, with the EA edition
and player page preserved in the record. A clearly labelled Soccerdonna fallback
is used only when the player is not represented in the EA SPORTS FC database.
"""

from __future__ import annotations

import json
from pathlib import Path

from acl_motion.persistence import atomic_write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_PATH = PROJECT_ROOT / "data/annotations/human/case_research_metadata_human.json"
HARMONIZATION_SOURCE = "project_taxonomy_harmonization:2026-08-31"


# player: (date of birth, preferred foot, injured side, Soccerdonna URL,
#          preferred-foot source, preferred-foot URL, league, competition, team)
PLAYER_METADATA = {
    "Beth Mead": ("1995-05-09", "right", "right", "https://www.soccerdonna.de/en/beth-mead/profil/spieler_12385.html", "Soccerdonna", "https://www.soccerdonna.de/en/beth-mead/profil/spieler_12385.html", "Women's Super League", "Women's Super League", "Arsenal"),
    "Vivianne Miedema": ("1996-07-15", "right", "left", "https://www.soccerdonna.de/en/vivianne-miedema/profil/spieler_9241.html", "Soccerdonna", "https://www.soccerdonna.de/en/vivianne-miedema/profil/spieler_9241.html", "Women's Super League", "UEFA Women's Champions League", "Arsenal"),
    "Leah Williamson": ("1997-03-29", "right", "right", "https://www.soccerdonna.de/en/leah-williamson/profil/spieler_22175.html", "Soccerdonna", "https://www.soccerdonna.de/en/leah-williamson/profil/spieler_22175.html", "Women's Super League", "Women's Super League", "Arsenal"),
    "Jordan Nobbs": ("1992-12-08", "right", "left", "https://www.soccerdonna.de/en/jordan-nobbs/profil/spieler_2585.html", "Soccerdonna", "https://www.soccerdonna.de/en/jordan-nobbs/profil/spieler_2585.html", "Women's Super League", "Women's Super League", "Arsenal"),
    "Chloe Kelly": ("1998-01-15", "right", "right", "https://www.soccerdonna.de/en/kelly/profil/spieler_22176.html", "EA SPORTS FC 26", "https://www.ea.com/cs/games/ea-sports-fc/ratings/player-ratings/chloe-kelly/257001", "Women's Super League", "Women's Super League", "Manchester City Women"),
    "Christen Press": ("1988-12-29", "right", "right", "https://www.soccerdonna.de/en/press/profil/spieler_8301.html", "EA SPORTS FC 26", "https://www.ea.com/de/games/ea-sports-fc/ratings/player-ratings/christen-press/226327", "National Women's Soccer League", "National Women's Soccer League", "Angel City FC"),
    "Delphine Cascarino": ("1997-02-05", "right", "right", "https://www.soccerdonna.de/en/delphine-cascarino/profil/spieler_17984.html", "Soccerdonna", "https://www.soccerdonna.de/en/delphine-cascarino/profil/spieler_17984.html", "Division 1 Féminine", "Division 1 Féminine", "Olympique Lyonnais"),
    "Ellie Carpenter": ("2000-04-28", "right", "right", "https://www.soccerdonna.de/en/ellie-carpenter/profil/spieler_30942.html", "Soccerdonna", "https://www.soccerdonna.de/en/ellie-carpenter/profil/spieler_30942.html", "Division 1 Féminine", "UEFA Women's Champions League", "Olympique Lyonnais"),
    "Kirsten van de Westeringh": ("2001-06-06", "right", "right", "https://www.soccerdonna.de/en/kirsten-van-de-westeringh/profil/spieler_35149.html", "Soccerdonna", "https://www.soccerdonna.de/en/kirsten-van-de-westeringh/profil/spieler_35149.html", "Vrouwen Eredivisie", "Vrouwen Eredivisie", "Feyenoord"),
    "Holly McNamara": ("2003-01-23", "right", "right", "https://www.soccerdonna.de/en/holly-mcnamara/profil/spieler_64222.html", "Soccerdonna", "https://www.soccerdonna.de/en/holly-mcnamara/profil/spieler_64222.html", "A-League Women", "A-League Women", "Melbourne City"),
    "Caroline Weir": ("1995-06-20", "left", "left", "https://www.soccerdonna.de/en/caroline-weir/profil/spieler_10461.html", "Soccerdonna", "https://www.soccerdonna.de/en/caroline-weir/profil/spieler_10461.html", "Liga F", "UEFA Women's Nations League", "Scotland"),
    "Mary Fowler": ("2003-02-14", "right", "right", "https://www.soccerdonna.de/en/mary-fowler/profil/spieler_39964.html", "Soccerdonna", "https://www.soccerdonna.de/en/mary-fowler/profil/spieler_39964.html", "Women's Super League", "Women's FA Cup", "Manchester City Women"),
    "Carmen Menayo": ("1998-04-14", "left", "right", "https://www.soccerdonna.de/en/carmen-menayo/profil/spieler_22162.html", "Soccerdonna", "https://www.soccerdonna.de/en/carmen-menayo/profil/spieler_22162.html", "Liga F", "Liga F", "Atlético de Madrid"),
    "Grace Wisnewski": ("2002-06-28", "right", "left", "https://www.soccerdonna.de/en/grace-wisnewski/profil/spieler_37811.html", "Soccerdonna", "https://www.soccerdonna.de/en/grace-wisnewski/profil/spieler_37811.html", "A-League Women", "A-League Women", "Wellington Phoenix"),
    "Ludmila da Silva": ("1994-12-01", "right", "right", "https://www.soccerdonna.de/en/ludmila/profil/spieler_24252.html", "Soccerdonna", "https://www.soccerdonna.de/en/ludmila/profil/spieler_24252.html", "Liga F", "Liga F", "Atlético de Madrid"),
    "Laura Wienroither": ("1999-01-13", "right", "left", "https://www.soccerdonna.de/en/laura-wienroither/profil/spieler_23374.html", "Soccerdonna", "https://www.soccerdonna.de/en/laura-wienroither/profil/spieler_23374.html", "Women's Super League", "UEFA Women's Champions League", "Arsenal"),
    "Aurora Galli": ("1996-12-13", "right", "right", "https://www.soccerdonna.de/en/aurora-galli/profil/spieler_22073.html", "Soccerdonna", "https://www.soccerdonna.de/en/aurora-galli/profil/spieler_22073.html", "Women's Super League", "Women's Super League", "Everton"),
    "Lena Oberdorf": ("2001-12-19", "right", "right", "https://www.soccerdonna.de/en/lena-oberdorf/profil/spieler_31601.html", "EA SPORTS FC 26", "https://www.ea.com/cs/games/ea-sports-fc/ratings/player-ratings/lena-oberdorf/248717", "Frauen-Bundesliga", "Frauen-Bundesliga", "FC Bayern München"),
    "Inma Gabarro": ("2002-11-05", "right", "left", "https://www.soccerdonna.de/en/gabarro/profil/spieler_43216.html", "Soccerdonna", "https://www.soccerdonna.de/en/gabarro/profil/spieler_43216.html", "Women's Super League", "Women's Super League", "Everton"),
    "Marie Höbinger": ("2001-07-01", "left", "left", "https://www.soccerdonna.de/en/marie-hoebinger/profil/spieler_30366.html", "EA SPORTS FC 26", "https://www.ea.com/en/games/ea-sports-fc/ratings/player-ratings/marie-hobinger/265019", "Women's Super League", "Women's Super League", "Liverpool Women"),
    "Charlotte Newsham": ("2000-05-14", "right", "right", "https://www.soccerdonna.de/en/charlotte-newsham/profil/spieler_40814.html", "Soccerdonna", "https://www.soccerdonna.de/en/charlotte-newsham/profil/spieler_40814.html", "Women's Super League 2", "Women's Super League 2", "Charlton Athletic Women"),
    "Sofie Lundgaard": ("2002-05-29", "right", "left", "https://www.soccerdonna.de/en/sofie-lundgaard/profil/spieler_36990.html", "Soccerdonna", "https://www.soccerdonna.de/en/sofie-lundgaard/profil/spieler_36990.html", "Women's Super League", "Women's Super League", "Liverpool Women"),
    "Kosovare Asllani": ("1989-07-29", "right", "left", "https://www.soccerdonna.de/en/kosovare-asllani/profil/spieler_68.html", "Soccerdonna", "https://www.soccerdonna.de/en/kosovare-asllani/profil/spieler_68.html", "Women's Super League", "Women's Super League", "London City Lionesses"),
    "Tierna Davidson": ("1998-09-19", "left", "left", "https://www.soccerdonna.de/en/davidson/profil/spieler_33936.html", "Soccerdonna", "https://www.soccerdonna.de/en/davidson/profil/spieler_33936.html", "National Women's Soccer League", "National Women's Soccer League", "Gotham FC"),
    "Malou Marcetto": ("2003-04-16", "right", "right", "https://www.soccerdonna.de/en/malou-marcetto/profil/spieler_37423.html", "Soccerdonna", "https://www.soccerdonna.de/en/malou-marcetto/profil/spieler_37423.html", "Women's Super League", "Women's Super League", "London City Lionesses"),
    "Kayla Duran": ("1999-09-29", "right", "right", "https://www.soccerdonna.de/en/kayla-duran/profil.html/spieler_110390", "Soccerdonna", "https://www.soccerdonna.de/en/kayla-duran/profil.html/spieler_110390", "National Women's Soccer League", "National Women's Soccer League", "Gotham FC"),
    "Alana Cook": ("1997-04-11", "right", "left", "https://www.soccerdonna.de/en/cook/profil/spieler_38784.html", "Soccerdonna", "https://www.soccerdonna.de/en/cook/profil/spieler_38784.html", "National Women's Soccer League", "National Women's Soccer League", "Kansas City Current"),
    "Julie Dufour": ("2001-05-29", "right", "left", "https://www.soccerdonna.de/en/julie-dufour/profil/spieler_33453.html", "Soccerdonna", "https://www.soccerdonna.de/en/julie-dufour/profil/spieler_33453.html", "National Women's Soccer League", "National Women's Soccer League", "Portland Thorns FC"),
    "Sophie Schmidt": ("1988-06-28", "right", "left", "https://www.soccerdonna.de/en/sophie-schmidt/profil/spieler_1700.html", "Soccerdonna", "https://www.soccerdonna.de/en/sophie-schmidt/profil/spieler_1700.html", "National Women's Soccer League", "National Women's Soccer League", "Houston Dash"),
    "Caiya Hanks": ("2004-09-26", "right", "left", "https://www.soccerdonna.de/de/caiya-hanks/profil/spieler_107515.html", "EA SPORTS FC 27", "https://www.ea.com/games/ea-sports-fc/ratings/player-ratings/caiya-hanks/77865", "National Women's Soccer League", "National Women's Soccer League", "Portland Thorns FC"),
    "Midge Purce": ("1995-09-18", "right", "left", "https://www.soccerdonna.de/en/midge-purce/profil/spieler_17926.html", "Soccerdonna", "https://www.soccerdonna.de/en/midge-purce/profil/spieler_17926.html", "National Women's Soccer League", "National Women's Soccer League", "Gotham FC"),
    "Alex Loera": ("1999-06-19", "right", "right", "https://www.soccerdonna.de/en/loera/profil/spieler_67462.html", "Soccerdonna", "https://www.soccerdonna.de/en/loera/profil/spieler_67462.html", "National Women's Soccer League", "National Women's Soccer League", "Bay FC"),
    "Andi Sullivan": ("1995-12-20", "right", "right", "https://www.soccerdonna.de/en/andi-sullivan/profil/spieler_17930.html", "Soccerdonna", "https://www.soccerdonna.de/en/andi-sullivan/profil/spieler_17930.html", "National Women's Soccer League", "National Women's Soccer League", "Washington Spirit"),
    "Gabrielle Robinson": ("2000-06-18", "right", "right", "https://www.soccerdonna.de/en/gabrielle-robinson/leistungsdaten/spieler_84607_2025.html", "Soccerdonna", "https://www.soccerdonna.de/en/gabrielle-robinson/leistungsdaten/spieler_84607_2025.html", "National Women's Soccer League", "National Women's Soccer League", "Kansas City Current"),
    "Cloe Lacasse": ("1993-07-07", "right", "left", "https://www.soccerdonna.de/en/cloe-lacasse/profil/spieler_34363.html", "EA SPORTS FC 26", "https://www.ea.com/de/games/ea-sports-fc/ratings/player-ratings/cloe-lacasse/265571", "National Women's Soccer League", "National Women's Soccer League", "Utah Royals FC"),
    "Cloé Lacasse": ("1993-07-07", "right", "left", "https://www.soccerdonna.de/en/cloe-lacasse/profil/spieler_34363.html", "EA SPORTS FC 26", "https://www.ea.com/de/games/ea-sports-fc/ratings/player-ratings/cloe-lacasse/265571", "National Women's Soccer League", "National Women's Soccer League", "Utah Royals FC"),
    "Alex Pfeiffer": ("2007-11-26", "right", "right", "https://www.soccerdonna.de/en/alex-pfeiffer/profil/spieler_74052.html", "EA SPORTS FC 26", "https://www.ea.com/en/games/ea-sports-fc/ratings/player-ratings/alex-pfeiffer/70490", "National Women's Soccer League", "National Women's Soccer League", "Kansas City Current"),
}


# player: (preferred foot, EA SPORTS FC edition, official player URL)
#
# These entries represent a full-cohort EA SPORTS FC audit, not just the players
# whose Soccerdonna profiles were ambiguous. Jordan Nobbs last appears in FC 25;
# that archived value is linked through a database mirror because EA's current
# ratings site no longer exposes her player page.
EAFC_PREFERRED_FOOT = {
    "Beth Mead": ("right", "EA SPORTS FC 26", "https://www.ea.com/fi/games/ea-sports-fc/ratings/player-ratings/beth-mead/245802"),
    "Vivianne Miedema": ("right", "EA SPORTS FC 26", "https://www.ea.com/id/games/ea-sports-fc/ratings/player-ratings/vivianne-miedema/233746"),
    "Leah Williamson": ("right", "EA SPORTS FC 26", "https://www.ea.com/nl/games/ea-sports-fc/ratings/player-ratings/leah-williamson/246426"),
    "Jordan Nobbs": ("right", "EA SPORTS FC 25 (FIFPlay mirror)", "https://www.fifplay.com/fc-25/players/227257/jordan-nobbs/"),
    "Chloe Kelly": ("right", "EA SPORTS FC 26", "https://www.ea.com/cs/games/ea-sports-fc/ratings/player-ratings/chloe-kelly/257001"),
    "Christen Press": ("right", "EA SPORTS FC 26", "https://www.ea.com/de/games/ea-sports-fc/ratings/player-ratings/christen-press/226327"),
    "Delphine Cascarino": ("right", "EA SPORTS FC 26", "https://careers.ea.com/de/games/ea-sports-fc/ratings/player-ratings/delphine-cascarino/232202"),
    "Ellie Carpenter": ("right", "EA SPORTS FC 26", "https://careers.ea.com/cs/games/ea-sports-fc/ratings/player-ratings/ellie-carpenter/240030"),
    "Holly McNamara": ("right", "EA SPORTS FC 27", "https://www.ea.com/nl/games/ea-sports-fc/ratings/player-ratings/holly-mc-namara/266242"),
    "Caroline Weir": ("left", "EA SPORTS FC 27", "https://www.ea.com/de/games/ea-sports-fc/ratings/player-ratings/caroline-weir/245879"),
    "Mary Fowler": ("left", "EA SPORTS FC 26", "https://www.ea.com/de/games/ea-sports-fc/ratings/player-ratings/mary-fowler/248799"),
    "Carmen Menayo": ("left", "EA SPORTS FC 26", "https://www.ea.com/fi/games/ea-sports-fc/ratings/player-ratings/carmen-menayo/272112"),
    "Ludmila da Silva": ("right", "EA SPORTS FC 27", "https://www.ea.com/es/games/ea-sports-fc/ratings/player-ratings/ludmila/241023"),
    "Laura Wienroither": ("right", "EA SPORTS FC 26", "https://www.ea.com/de/games/ea-sports-fc/ratings/player-ratings/laura-wienroither/264996"),
    "Aurora Galli": ("right", "EA SPORTS FC 26", "https://www.ea.com/cs/games/ea-sports-fc/ratings/player-ratings/aurora-galli/245443"),
    "Lena Oberdorf": ("right", "EA SPORTS FC 26", "https://www.ea.com/cs/games/ea-sports-fc/ratings/player-ratings/lena-oberdorf/248717"),
    "Inma Gabarro": ("right", "EA SPORTS FC 26", "https://www.ea.com/nl/games/ea-sports-fc/ratings/player-ratings/inma-gabarro/272056"),
    "Marie Höbinger": ("left", "EA SPORTS FC 26", "https://www.ea.com/en/games/ea-sports-fc/ratings/player-ratings/marie-hobinger/265019"),
    "Charlotte Newsham": ("right", "EA SPORTS FC 27", "https://www.ea.com/es/games/ea-sports-fc/ratings/player-ratings/charlotte-newsham/86135"),
    "Sofie Lundgaard": ("right", "EA SPORTS FC 26", "https://www.ea.com/nl/games/ea-sports-fc/ratings/player-ratings/sofie-lundgaard/273050"),
    "Kosovare Asllani": ("right", "EA SPORTS FC 26", "https://www.ea.com/de/games/ea-sports-fc/ratings/player-ratings/kosovare-asllani/226987"),
    "Tierna Davidson": ("left", "EA SPORTS FC 26", "https://www.ea.com/it/games/ea-sports-fc/ratings/player-ratings/tierna-davidson/244045"),
    "Malou Marcetto": ("right", "EA SPORTS FC 26", "https://www.ea.com/ro/games/ea-sports-fc/ratings/player-ratings/malou-marcetto/276771"),
    "Kayla Duran": ("right", "EA SPORTS FC 27", "https://www.ea.com/es/games/ea-sports-fc/ratings/player-ratings/kayla-duran/80331"),
    "Alana Cook": ("right", "EA SPORTS FC 27", "https://www.ea.com/es-mx/games/ea-sports-fc/ratings/player-ratings/alana-cook/264011"),
    "Julie Dufour": ("right", "EA SPORTS FC 26", "https://www.ea.com/es/games/ea-sports-fc/ratings/player-ratings/julie-dufour/265860"),
    "Sophie Schmidt": ("right", "EA SPORTS FC 26", "https://www.ea.com/nl/games/ea-sports-fc/ratings/player-ratings/sophie-schmidt/227405"),
    "Caiya Hanks": ("right", "EA SPORTS FC 27", "https://www.ea.com/games/ea-sports-fc/ratings/player-ratings/caiya-hanks/77865"),
    "Midge Purce": ("right", "EA SPORTS FC 26", "https://www.ea.com/th/games/ea-sports-fc/ratings/player-ratings/midge-purce/243769"),
    "Alex Loera": ("right", "EA SPORTS FC 26", "https://www.ea.com/es/games/ea-sports-fc/ratings/player-ratings/alex-loera/267338"),
    "Andi Sullivan": ("right", "EA SPORTS FC 26", "https://www.ea.com/es/games/ea-sports-fc/ratings/player-ratings/andi-sullivan/243261"),
    "Gabrielle Robinson": ("right", "EA SPORTS FC 26", "https://www.ea.com/de/games/ea-sports-fc/ratings/player-ratings/gabrielle-robinson/273480"),
    "Cloe Lacasse": ("right", "EA SPORTS FC 26", "https://www.ea.com/de/games/ea-sports-fc/ratings/player-ratings/cloe-lacasse/265571"),
    "Cloé Lacasse": ("right", "EA SPORTS FC 26", "https://www.ea.com/de/games/ea-sports-fc/ratings/player-ratings/cloe-lacasse/265571"),
    "Alex Pfeiffer": ("right", "EA SPORTS FC 26", "https://www.ea.com/en/games/ea-sports-fc/ratings/player-ratings/alex-pfeiffer/70490"),
}


EAFC_NOT_LISTED = {
    "Kirsten van de Westeringh": "Vrouwen Eredivisie player not represented in the EA SPORTS FC ratings database",
    "Grace Wisnewski": "Player not represented in the EA SPORTS FC ratings database",
}


def main() -> None:
    payload = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    cases = payload.get("cases", {})
    updated = 0
    for record in cases.values():
        player_name = str(record.get("player_name", "")).strip()
        values = PLAYER_METADATA.get(player_name)
        if values is None:
            raise KeyError(f"No reviewed harmonization row for {player_name!r}")
        (
            date_of_birth,
            preferred_foot,
            injured_side,
            soccerdonna_url,
            preferred_foot_source,
            preferred_foot_url,
            league,
            competition,
            team,
        ) = values
        ea_fc = EAFC_PREFERRED_FOOT.get(player_name)
        if ea_fc is not None:
            preferred_foot, preferred_foot_source, preferred_foot_url = ea_fc
            ea_fc_audit_status = "verified"
            ea_fc_audit_note = "Preferred foot verified against the linked EA SPORTS FC player record."
        else:
            ea_fc_audit_status = "not_listed"
            ea_fc_audit_note = EAFC_NOT_LISTED[player_name]
            preferred_foot_source = "Soccerdonna fallback (not listed in EA SPORTS FC)"
            preferred_foot_url = soccerdonna_url
        record.update(
            {
                "date_of_birth": date_of_birth,
                "injured_side": injured_side,
                "preferred_foot": preferred_foot,
                "preferred_foot_source": preferred_foot_source,
                "preferred_foot_source_url": preferred_foot_url,
                "preferred_foot_knee_injured": preferred_foot == injured_side,
                "ea_fc_audit_status": ea_fc_audit_status,
                "ea_fc_audit_note": ea_fc_audit_note,
                "league": league,
                "competition": competition,
                "team": team,
                "harmonization_source": HARMONIZATION_SOURCE,
                "updated_at": "2026-08-31T00:00:00+00:00",
            }
        )
        if player_name == "Cloe Lacasse":
            record["player_name"] = "Cloé Lacasse"
        source_urls = dict(record.get("source_urls", {}))
        source_urls["soccerdonna_profile"] = soccerdonna_url
        source_urls["preferred_foot"] = preferred_foot_url
        record["source_urls"] = source_urls
        provenance = dict(record.get("field_provenance", {}))
        provenance.update(
            {
                "date_of_birth": "Soccerdonna player profile",
                "preferred_foot": preferred_foot_source,
                "preferred_foot_knee_injured": "derived:preferred_foot+injured_side",
                "league": HARMONIZATION_SOURCE,
                "competition": HARMONIZATION_SOURCE,
                "team": HARMONIZATION_SOURCE,
            }
        )
        record["field_provenance"] = provenance
        updated += 1

    payload["metadata_version"] = "case_research_metadata_v3_harmonized_taxonomy"
    payload["taxonomy"] = {
        "league_definition": "The player's domestic club league at the injury date.",
        "competition_definition": "The competition in which the injury event occurred.",
        "preferred_foot_knee_injured_definition": "True when the documented injured knee is on the same side as the sourced preferred foot.",
        "preferred_foot_source_policy": "EA SPORTS FC for every represented player; labelled Soccerdonna fallback only when EA SPORTS FC has no player record.",
        "ea_fc_verified_players": len(EAFC_PREFERRED_FOOT) - 2,
        "ea_fc_archived_mirror_players": 1,
        "ea_fc_not_listed_players": sorted(EAFC_NOT_LISTED),
        "league_values": sorted({values[6] for values in PLAYER_METADATA.values()}),
        "competition_values": sorted({values[7] for values in PLAYER_METADATA.values()}),
        "team_aliases_applied": {
            "Manchester City": "Manchester City Women",
            "Liverpool": "Liverpool Women",
            "Portland Thorns": "Portland Thorns FC",
            "Atlético Madrid": "Atlético de Madrid",
        },
    }
    atomic_write_json(METADATA_PATH, payload, trailing_newline=True)
    print(f"Updated {updated} case records in {METADATA_PATH}")


if __name__ == "__main__":
    main()
