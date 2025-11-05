import psycopg2

from src.information_extraction.information_extraction import InformationExtractor


class DataTransformer:

    def __init__(self,  input_folder="../../input",
                        image_name ="form_3_1A.png",
                        rerun_ocr=False,
                        dbname="image_processing",
                        user="postgres",
                        password="postgres",
                        host="localhost",
                        port="5432"
                            ):
        self.dbname = dbname
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.image_name = image_name
        self.conn = None
        self.info_extractor = InformationExtractor(input_folder=input_folder, input_image=image_name)
        self.info_extractor.run(rerun_ocr=rerun_ocr)
        self.data = self.info_extractor.extracted_data
        self.form_type = self.info_extractor.form_type.lower()



    def create_connect_to_db(self):
        try:
            self.conn  = psycopg2.connect(
                dbname=self.dbname,
                user=self.user,
                password=self.password,
                host=self.host,
                port=self.port
            )
            print("✅ Connection to database successful.")
        except psycopg2.Error as e:
            print("❌Error connecting to the database:", e)

    def create_forms_table_if_not_exists(self):
        self.create_connect_to_db()
        if self.conn is None:
            print("❌ Error connecting to the database")
            return False

        self.create_metadata_table()
        self.create_table_form1_tasks()
        self.create_table_form2_tasks()
        self.create_table_form3_tasks()
        return True


    def create_metadata_table(self):
        if self.conn is None:
            print("❌ Keine aktive Datenbankverbindung.")
            return

        create_table_sql = """
                           CREATE TABLE IF NOT EXISTS form_metadata( 
                               id SERIAL PRIMARY KEY, 
                               form_type VARCHAR (20),
                               image_name VARCHAR (255),
                               name TEXT,
                               traeger TEXT,
                               geburtsdatum TEXT,
                               fls_anzahl_bewilligt TEXT,
                               bescheid_von_bis TEXT,
                               nachweiszeitraum TEXT,
                               gesamt_fls_nachweiszeitraum TEXT,
                               arbeitsbereich TEXT,
                               bedarf_itp TEXT,
                               created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                               );
                           """

        try:
            with self.conn.cursor() as cur:
                cur.execute(create_table_sql)
                self.conn.commit()
                print("✅ Tabelle 'form_metadata' erfolgreich erstellt oder bereits vorhanden.")
        except psycopg2.Error as e:
            print("❌ Fehler beim Erstellen der Tabelle (form_metadata):", e)
            self.conn.rollback()


    def create_table_form1_tasks(self):
        if self.conn is None:
            print("❌ Keine aktive Datenbankverbindung.")
            return
        create_table_sql = """
                                CREATE TABLE IF NOT EXISTS taetigkeiten_form1 (
                                id SERIAL PRIMARY KEY,
                                form_id INTEGER REFERENCES form_metadata(id) ON DELETE CASCADE,
                                datum_uhrzeit TEXT,
                                dauer_fls TEXT,
                                allein BOOLEAN,
                                in_der_gruppe BOOLEAN,
                                mitarbeiter_geholfen BOOLEAN,
                                hilfe_zufriedenheit BOOLEAN,
                                unterschrift_lb TEXT,
                                zielbezogene_unterstuetzung TEXT,
                                leistung_abgenommen BOOLEAN,
                                leistung_bewertet BOOLEAN,
                                wenn_nicht_weil TEXT,
                                handzeichen_persnr TEXT
                                );
                        """
        try:
            with self.conn.cursor() as cur:
                cur.execute(create_table_sql)
                self.conn.commit()
                print("✅ Tabelle 'taetigkeiten_form1' erfolgreich erstellt oder bereits vorhanden.")
        except psycopg2.Error as e:
            print("❌ Fehler beim Erstellen der Tabelle (taetigkeiten_form1):", e)
            self.conn.rollback()


    def create_table_form2_tasks(self):
        if self.conn is None:
            print("❌ Keine aktive Datenbankverbindung.")
            return
        create_table_sql ="""
                                CREATE TABLE IF NOT EXISTS taetigkeiten_form2 (
                                id SERIAL PRIMARY KEY,
                                meta_id INTEGER REFERENCES form_metadata(id) ON DELETE CASCADE,
                                datum TEXT,
                                uhrzeit_von_bis TEXT,
                                leistung_abgenommen BOOLEAN,
                                leistung_bewertet BOOLEAN,
                                wenn_nicht_weil TEXT,
                                unterschrift_le TEXT,
                                leistung_zufriedenheit BOOLEAN,
                                unterschrift_lb TEXT
                                );
                          """
        try:
            with self.conn.cursor() as cur:
                cur.execute(create_table_sql)
                self.conn.commit()
                print("✅ Tabelle 'taetigkeiten_form2' erfolgreich erstellt oder bereits vorhanden.")
        except psycopg2.Error as e:
            print("❌ Fehler beim Erstellen der Tabelle (taetigkeiten_form2):", e)
            self.conn.rollback()

    def create_table_form3_tasks(self):
        if self.conn is None:
            print("❌ Keine aktive Datenbankverbindung.")
            return

        create_table_sql = """
                               CREATE TABLE IF NOT EXISTS taetigkeiten_form3 (
                                id SERIAL PRIMARY KEY,
                                meta_id INTEGER REFERENCES form_metadata(id) ON DELETE CASCADE,
                                datum TEXT,
                                fls TEXT,
                                leistung_abgenommen BOOLEAN,
                                leistung_bewertet BOOLEAN,
                                wenn_nicht_weil TEXT,
                                allein BOOLEAN,
                                gruppe BOOLEAN,
                                handzeichen_le TEXT,
                                leistung_zufriedenheit BOOLEAN
                            );
                        """
        try:
            with self.conn.cursor() as cur:
                cur.execute(create_table_sql)
                self.conn.commit()
                print("✅ Tabelle 'taetigkeiten_form3' erfolgreich erstellt oder bereits vorhanden.")
        except psycopg2.Error as e:
            print("❌ Fehler beim Erstellen der Tabelle (taetigkeiten_form3):", e)
            self.conn.rollback()

    def save_data_into_db(self):
        self.create_connect_to_db()
        if self.conn is None:
            print("❌ Keine aktive Datenbankverbindung.")
            return
        if self.data is None or len(self.data) < 1:
            print("❌ Ungültige Datenstruktur.")
            return

        meta_data, tasks = self.data
        try:
            with self.conn.cursor() as cur:
                insert_meta = """
                              INSERT INTO form_metadata (form_type, image_name, name, traeger, geburtsdatum, \
                                                         fls_anzahl_bewilligt, bescheid_von_bis, \
                                                         nachweiszeitraum, gesamt_fls_nachweiszeitraum, \
                                                         arbeitsbereich, bedarf_itp) \
                              VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id; \
                              """
                cur.execute(insert_meta, (
                    self.form_type,
                    self.image_name,
                    meta_data.get('Name, Vorname') or meta_data.get("Name Vorname"),
                    meta_data.get('Träger') or meta_data.get('traeger'),
                    meta_data.get('Geburtsdatum'),
                    meta_data.get(
                        'Anzahl bewilligter FLS bzw. ganz- tags/halbtags im Bewilligungszeitraum') or meta_data.get(
                        'Anzahl bewilligter FLS'),
                    meta_data.get("Bescheid von/bis") or meta_data.get("Bedarfszeitraum von/bis"),
                    meta_data.get("Nachweiszeitraum: (Monat/jahr)") or meta_data.get(
                        'Nachweiszeitraum (Monat/Jahr)'),
                    meta_data.get('Gesamt FLS im Nachweiszeitraum'),
                    meta_data.get('Arbeitsbereich'),
                    meta_data.get('Bedarf Laut ITP')
                ))
                meta_id = cur.fetchone()[0]

                if self.form_type.startswith("form_1"):
                    insert_task = """
                                  INSERT INTO taetigkeiten_form1 (form_id, datum_uhrzeit, dauer_fls, allein, \
                                                                  in_der_gruppe, \
                                                                  mitarbeiter_geholfen, hilfe_zufriedenheit, \
                                                                  unterschrift_lb, \
                                                                  zielbezogene_unterstuetzung, leistung_abgenommen, \
                                                                  leistung_bewertet, wenn_nicht_weil, \
                                                                  handzeichen_persnr) \
                                  VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s); \
                                  """
                    for task in tasks:
                        cur.execute(insert_task, (
                            meta_id,
                            task.get("am (Datum) um (Uhrzeit)"),
                            task.get("Dauer in FLS"),
                            task.get("Allein") is not None,
                            task.get("In der Gruppe") is not None,
                            task.get("Die Mitarbeiter haben mir geholfen.") is not None,
                            task.get("Ich war mit der Hilfe zufrieden.") is not None,
                            task.get("Unterschrift LB"),
                            task.get("Zielbezogene Unterstützungsleistung gem. Teilhabeplan, Maßnahmedarstellung und Ort der Leistungserbringung"),
                            task.get("Der Leistungsberechtigte hat die Leistung abgenommen und bewertet") is not None,
                            task.get("Der Leistungsberechtigte hat die Leistung abgenommen und bewertet") is not None,
                            task.get("Wenn nicht, weil"),
                            task.get("Handzeichen LE+ Pers.Nr.")
                        ))

                elif self.form_type.startswith("form_2"):
                    insert_task = """
                                  INSERT INTO taetigkeiten_form2 (meta_id, datum, uhrzeit_von_bis, leistung_abgenommen, \
                                                                  leistung_bewertet, wenn_nicht_weil, unterschrift_le, \
                                                                  leistung_zufriedenheit, unterschrift_lb) \
                                  VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s); \
                                  """
                    for task in tasks:
                        cur.execute(insert_task, (
                            meta_id,
                            task.get("Datum"),
                            task.get("Uhrzeit von-bis"),
                            task.get("abgenommen") is not None,
                            task.get("bewertet") is not None,
                            task.get("Wenn nicht, weil"),
                            task.get("Unterschrift LE"),
                            task.get("Ich bin mit der Leistung zufrieden") is not None,
                            task.get("Unterschrift LB")
                        ))


                elif self.form_type.startswith("form_3"):

                    insert_task = """

                                  INSERT INTO taetigkeiten_form3 (meta_id, datum, fls, leistung_abgenommen, \
                                                                  leistung_bewertet, wenn_nicht_weil, allein, gruppe, \
                                                                  handzeichen_le, leistung_zufriedenheit) \
                                  VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s); \
                                  """

                    for task in tasks:
                        cur.execute(insert_task, (
                            meta_id,
                            task.get("Datum"),
                            task.get("FLS"),
                            task.get("abgenommen") != "ja",
                            task.get("bewertet") is not None,
                            task.get("Wenn nicht, weil:"),
                            task.get("allein") is not None,
                            task.get("Gruppe") is not None,
                            task.get("Handzeichen LE"),
                            task.get("Der Mitarbeiter hat mir geholfen/ Ich war mit der Hilfe: zufrieden") is not None
                        ))

                self.conn.commit()

                print(f"✅ Daten erfolgreich in die Datenbank gespeichert. Formtyp: {self.form_type}")
        except psycopg2.Error as e:
            print("❌ Fehler beim Speichern der Daten:", e)
            self.conn.rollback()