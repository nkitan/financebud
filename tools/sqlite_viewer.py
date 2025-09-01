#!/usr/bin/env python3
"""
Simple SQLite viewer/editor GUI
--------------------------------

Features:
- Open a SQLite file
- List tables
- View rows (first 1000) in a table
- Edit, add, delete rows using a simple form
- Run ad-hoc SELECT queries

This tool uses only the Python standard library (tkinter + sqlite3).

Usage:
    python tools/sqlite_viewer.py [path/to/database.db]

"""
import os
import sqlite3
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import logging

# Setup file logger for the sqlite viewer
LOG_PATH = os.path.join(os.path.dirname(__file__), "sql_viewer.log")
logger = logging.getLogger("sqlite_viewer")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(LOG_PATH)
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)


DEFAULT_LIMIT = 1000


class SQLiteViewer(tk.Tk):
    def __init__(self, db_path=None):
        super().__init__()
        self.title("SQLite Viewer / Editor")
        self.geometry("1000x600")

        self.db_path = db_path or os.path.join(os.getcwd(), "financial_data.db")
        self.conn = None

        # internal state
        self._current_columns_types = []
        self._current_pk_cols = []
        self._current_offset = 0
        # map treeview item id -> primary key value for that row
        self._iid_to_pk = {}

        self._build_ui()

        if self.db_path and os.path.exists(self.db_path):
            self.open_database(self.db_path)

    def _build_ui(self):
        # Top frame: file controls and table selector
        top = ttk.Frame(self)
        top.pack(fill=tk.X, padx=6, pady=6)
    
        self.open_btn = ttk.Button(top, text="Open DB", command=self.on_open_db)
        self.open_btn.pack(side=tk.LEFT)
    
        self.db_label = ttk.Label(top, text="No database loaded")
        self.db_label.pack(side=tk.LEFT, padx=8)
    
        ttk.Label(top, text="Tables:").pack(side=tk.LEFT, padx=(12, 4))
        self.table_cb = ttk.Combobox(top, values=[], state="readonly", width=40)
        self.table_cb.pack(side=tk.LEFT)
        self.table_cb.bind("<<ComboboxSelected>>", lambda e: self.load_table())
    
        refresh_btn = ttk.Button(top, text="Refresh Tables", command=self.refresh_tables)
        refresh_btn.pack(side=tk.LEFT, padx=6)
    
        # Filter and pagination
        filter_frame = ttk.Frame(self)
        filter_frame.pack(fill=tk.X, padx=6)
    
        ttk.Label(filter_frame, text="Filter (WHERE clause):").pack(side=tk.LEFT)
        self.filter_entry = ttk.Entry(filter_frame, width=60)
        self.filter_entry.pack(side=tk.LEFT, padx=6)
        ttk.Button(filter_frame, text="Apply", command=self.on_apply_filter).pack(side=tk.LEFT)
    
        # Pagination controls
        pagination = ttk.Frame(self)
        pagination.pack(fill=tk.X, padx=6, pady=(4,0))
        ttk.Label(pagination, text="Page size:").pack(side=tk.LEFT)
        # Combobox expects strings; use StringVar and convert when reading
        self.page_size_var = tk.StringVar(value='100')
        self.page_size_cb = ttk.Combobox(pagination, values=("50","100","200","500"), textvariable=self.page_size_var, width=6)
        self.page_size_cb.pack(side=tk.LEFT, padx=4)
        ttk.Button(pagination, text="Prev", command=self.on_prev_page).pack(side=tk.LEFT, padx=4)
        ttk.Button(pagination, text="Next", command=self.on_next_page).pack(side=tk.LEFT, padx=4)
        self.page_label = ttk.Label(pagination, text="Page 1")
        self.page_label.pack(side=tk.LEFT, padx=8)
    
        # Middle frame: treeview for rows
        mid = ttk.Frame(self)
        mid.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0,6))
    
        self.tree = ttk.Treeview(mid, columns=(), show="headings")
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.tree.bind("<Double-1>", self.on_edit_cell)
    
        vsb = ttk.Scrollbar(mid, orient="vertical", command=self.tree.yview)
        vsb.pack(side=tk.LEFT, fill=tk.Y)
        self.tree.configure(yscrollcommand=vsb.set)
    
        # Right side: actions
        right = ttk.Frame(self)
        right.pack(fill=tk.X, padx=6, pady=6)
    
        btn_frame = ttk.Frame(right)
        btn_frame.pack(side=tk.TOP, anchor=tk.W)
    
        ttk.Button(btn_frame, text="Add Row", command=self.on_add_row).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Edit Row", command=self.on_edit_row).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Delete Row", command=self.on_delete_row).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Run Query", command=self.on_run_query).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Refresh Rows", command=self.load_table).pack(side=tk.LEFT, padx=4)
    
        # Status bar
        self.status = ttk.Label(self, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status.pack(fill=tk.X, side=tk.BOTTOM)

    # --- DB handling
    def open_database(self, path):
        try:
            if self.conn:
                self.conn.close()
            self.conn = sqlite3.connect(path)
            self.conn.row_factory = sqlite3.Row
            self.db_path = path
            self.db_label.config(text=os.path.basename(path))
            self.refresh_tables()
            self.status.config(text=f"Opened {path}")
            logger.info(f"Opened database: {path}")
        except Exception as e:
            logger.exception(f"Failed to open database: {path}")
            messagebox.showerror("Open DB", f"Failed to open database: {e}")

    def on_open_db(self):
        p = filedialog.askopenfilename(title="Select SQLite DB", filetypes=[("SQLite DB", "*.db *.sqlite3 *.sqlite"), ("All files", "*")])
        if p:
            self.open_database(p)

    def refresh_tables(self):
        if not self.conn:
            return
        cur = self.conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")
        tables = [r[0] for r in cur.fetchall()]
        self.table_cb['values'] = tables
        if tables:
            # auto-select first if none selected
            if not self.table_cb.get():
                self.table_cb.set(tables[0])
                # reset pagination and filter
                self._current_offset = 0
                self.filter_entry.delete(0, tk.END)
                self.load_table()

    def _quote_identifier(self, name: str) -> str:
        """Return a safely quoted identifier for use in SQL statements."""
        return '"' + str(name).replace('"', '""') + '"'

    # --- Table viewing
    def load_table(self, *_):
        table = self.table_cb.get()
        if not table or not self.conn:
            return
        cur = self.conn.cursor()

        # Get schema info via PRAGMA, but handle errors gracefully
        quoted_table = self._quote_identifier(table)
        try:
            cur.execute(f"PRAGMA table_info({quoted_table})")
            cols_info = cur.fetchall()
            columns = [c[1] for c in cols_info]
            types = [c[2] for c in cols_info]
            pk_cols = [c[1] for c in cols_info if c[5] != 0]
        except Exception as e:
            # PRAGMA failed (odd schema name); fallback to attempting SELECT * LIMIT 1
            try:
                cur.execute(f"SELECT * FROM {quoted_table} LIMIT 1")
                cols = [d[0] for d in cur.description]
                columns = cols
                types = ['TEXT'] * len(cols)
                pk_cols = ["rowid"]
            except Exception as ex:
                # Give a non-blocking error and return
                self.status.config(text=f"Error getting schema for {table}: {ex}")
                return

        if not pk_cols:
            pk_cols = ["rowid"]

        # store schema info for safer editing
        self._current_columns_types = list(zip(columns, types))
        self._current_pk_cols = pk_cols

        # Set up treeview
        self._iid_to_pk.clear()
        self.tree.delete(*self.tree.get_children())
        # If there are no columns (rare), show message and stop
        if not columns:
            self.status.config(text=f"Table '{table}' has no columns to display")
            return

        self.tree['columns'] = columns
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=120, stretch=True)

        # Query rows with limit/offset and optional WHERE filter. If the WHERE filter
        # causes an error, retry without it to show rows and allow the user to refine the filter.
        page_size = int(self.page_size_var.get() or 100)
        limit = page_size
        offset = int(self._current_offset or 0)
        where_clause = self.filter_entry.get().strip()

        base_sql = f"SELECT rowid, * FROM {quoted_table}"
        sql = base_sql
        if where_clause:
            sql = f"{base_sql} WHERE {where_clause}"
        sql = f"{sql} LIMIT {limit} OFFSET {offset}"

        try:
            cur.execute(sql)
            rows = cur.fetchall()
        except Exception as e:
            # retry without where clause
            try:
                fallback_sql = f"{base_sql} LIMIT {limit} OFFSET {offset}"
                cur.execute(fallback_sql)
                rows = cur.fetchall()
                self.status.config(text=f"Filter error - showing unfiltered rows: {e}")
            except Exception as ex:
                self.status.config(text=f"Error loading rows for {table}: {ex}")
                return

        import uuid
        for i, row in enumerate(rows):
            # row is sqlite3.Row; convert to tuple of values excluding the first (rowid)
            values = [row[col] for col in columns]
            # determine primary key value
            if pk_cols and pk_cols != ["rowid"]:
                try:
                    pk_val = row[pk_cols[0]]
                except Exception:
                    pk_val = None
            else:
                # use the sqlite rowid returned by the query
                pk_val = row['rowid'] if 'rowid' in row.keys() else None

            # generate a unique tree iid to avoid collisions across reloads
            iid = uuid.uuid4().hex
            self._iid_to_pk[iid] = pk_val
            try:
                self.tree.insert('', tk.END, iid=iid, values=values)
            except Exception:
                # skip problematic row inserts but continue
                continue

        # update page label
        self.page_label.config(text=f"Page {(offset//limit)+1}")

        self.status.config(text=f"Loaded {len(rows)} rows from {table} (showing up to {limit})")

    # --- Row operations
    def _get_table_and_schema(self):
        table = self.table_cb.get()
        if not table:
            messagebox.showwarning("No table", "Please select a table first")
            return None, None, None
        # use cached schema when available
        if self._current_columns_types:
            columns = [c for c, _ in self._current_columns_types]
            pk_cols = self._current_pk_cols or ["rowid"]
            return table, columns, pk_cols

        cur = self.conn.cursor()
        cur.execute(f"PRAGMA table_info({table})")
        cols_info = cur.fetchall()
        columns = [c[1] for c in cols_info]
        pk_cols = [c[1] for c in cols_info if c[5] != 0]
        if not pk_cols:
            pk_cols = ["rowid"]
        return table, columns, pk_cols

    def on_add_row(self):
        table, columns, pk = self._get_table_and_schema()
        if not table:
            return
        values = self._prompt_row_values(columns, initial=None, title=f"Add row to {table}")
        if values is None:
            return
        try:
            placeholders = ','.join('?' for _ in columns)
            cur = self.conn.cursor()
            quoted_table = self._quote_identifier(table)
            quoted_cols = ', '.join(self._quote_identifier(c) for c in columns)
            cur.execute(f"INSERT INTO {quoted_table} ({quoted_cols}) VALUES ({placeholders})", values)
            self.conn.commit()
            self.load_table()
            self.status.config(text=f"Inserted row into {table}")
            logger.info(f"Inserted row into {table}: {values}")
        except Exception as e:
            logger.exception(f"Failed to insert row into {table}: {e}")
            messagebox.showerror("Insert Row", f"Failed to insert row: {e}")

    def on_edit_row(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showwarning("Select row", "Please select a row to edit")
            return
        rowid = sel[0]
        table, columns, pk = self._get_table_and_schema()
        if not table:
            return
        # fetch current values
        cur = self.conn.cursor()
        try:
            quoted_table = self._quote_identifier(table)
            if pk == ["rowid"]:
                pk_val = self._iid_to_pk.get(rowid)
                try:
                    cur.execute(f"SELECT rowid, * FROM {quoted_table} WHERE rowid = ?", (pk_val,))
                except Exception as e:
                    logger.exception(f"SELECT by rowid failed for {table} rowid={pk_val}: {e}")
                    raise
            else:
                # assume single pk
                pk_val = self._iid_to_pk.get(rowid)
                quoted_pk = self._quote_identifier(pk[0])
                try:
                    cur.execute(f"SELECT * FROM {quoted_table} WHERE {quoted_pk} = ?", (pk_val,))
                except Exception as e:
                    logger.exception(f"SELECT by pk failed for {table} {pk[0]}={pk_val}: {e}")
                    raise
            row = cur.fetchone()
            if not row:
                messagebox.showerror("Edit Row", "Could not find selected row in database")
                return
            current = [row[c] for c in columns]
            new_values = self._prompt_row_values(columns, initial=current, title=f"Edit row {rowid} in {table}")
            if new_values is None:
                return

            # Build UPDATE
            set_clause = ','.join(f"{c} = ?" for c in columns)
            if pk == ["rowid"]:
                where_clause = "rowid = ?"
                params = list(new_values) + [pk_val]
            else:
                where_clause = f"{pk[0]} = ?"
                params = list(new_values) + [row[pk[0]]]

            quoted_table = self._quote_identifier(table)
            try:
                cur.execute(f"UPDATE {quoted_table} SET {set_clause} WHERE {where_clause}", params)
                self.conn.commit()
                self.load_table()
                self.status.config(text=f"Updated row {rowid} in {table}")
                logger.info(f"Updated row {rowid} in {table}: {params}")
            except Exception as e:
                logger.exception(f"Failed to update row {rowid} in {table}: {e}")
                raise
        except Exception as e:
            messagebox.showerror("Edit Row", f"Failed to update row: {e}")

    def on_delete_row(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showwarning("Select row", "Please select a row to delete")
            return
        if not messagebox.askyesno("Delete", "Delete selected row(s)? This cannot be undone."):
            return
        table, columns, pk = self._get_table_and_schema()
        if not table:
            return
        cur = self.conn.cursor()
        errors = []
        for iid in sel:
            try:
                pk_val = self._iid_to_pk.get(iid)
                quoted_table = self._quote_identifier(table)
                if pk == ["rowid"]:
                    cur.execute(f"DELETE FROM {quoted_table} WHERE rowid = ?", (pk_val,))
                else:
                    quoted_pk = self._quote_identifier(pk[0])
                    cur.execute(f"DELETE FROM {quoted_table} WHERE {quoted_pk} = ?", (pk_val,))
            except Exception as e:
                logger.exception(f"Failed to delete iid={iid} from {table}: {e}")
                errors.append(str(e))
        self.conn.commit()
        self.load_table()
        if errors:
            messagebox.showerror("Delete", "Some deletes failed:\n" + "\n".join(errors))
        else:
            self.status.config(text=f"Deleted {len(sel)} row(s) from {table}")

    def on_edit_cell(self, event):
        # double click -> edit selected row
        self.on_edit_row()

    def _prompt_row_values(self, columns, initial=None, title="Edit"):
        dlg = tk.Toplevel(self)
        dlg.title(title)
        entries = {}
        # try to use cached column types to provide hints and protect PK fields
        types_map = {c: t for c, t in (self._current_columns_types or [])}
        for i, col in enumerate(columns):
            ttk.Label(dlg, text=f"{col} ({types_map.get(col, 'TEXT')})").grid(row=i, column=0, sticky=tk.W, padx=6, pady=4)
            e = ttk.Entry(dlg, width=80)
            e.grid(row=i, column=1, sticky=tk.W, padx=6, pady=4)
            if initial is not None:
                val = initial[i]
                e.insert(0, "" if val is None else str(val))
            # disable PK columns from editing when editing existing row
            if self._current_pk_cols and col in self._current_pk_cols and initial is not None:
                e.config(state='disabled')
            entries[col] = e

        result = {'ok': False}

        def on_ok():
            result['ok'] = True
            dlg.destroy()

        def on_cancel():
            dlg.destroy()

        btns = ttk.Frame(dlg)
        btns.grid(row=len(columns), column=0, columnspan=2, pady=8)
        ttk.Button(btns, text="OK", command=on_ok).pack(side=tk.LEFT, padx=6)
        ttk.Button(btns, text="Cancel", command=on_cancel).pack(side=tk.LEFT, padx=6)

        dlg.transient(self)
        dlg.grab_set()
        self.wait_window(dlg)

        if not result['ok']:
            return None

        values = [entries[c].get() or None for c in columns]
        # Convert empty strings to None
        values = [None if v == '' else v for v in values]
        return values

    # Pagination/filter helpers
    def on_apply_filter(self):
        self._current_offset = 0
        self.load_table()

    def on_next_page(self):
        page_size = int(self.page_size_var.get() or 100)
        self._current_offset = int(self._current_offset or 0) + page_size
        self.load_table()

    def on_prev_page(self):
        page_size = int(self.page_size_var.get() or 100)
        self._current_offset = max(0, int(self._current_offset or 0) - page_size)
        self.load_table()

    def on_run_query(self):
        q = simpledialog.askstring("Run query", "Enter SELECT query to run:", parent=self)
        if not q:
            return
        if not q.strip().lower().startswith('select'):
            messagebox.showwarning("Only SELECT", "For safety this tool only runs SELECT queries from the Run Query dialog.")
            return
        try:
            cur = self.conn.cursor()
            cur.execute(q)
            rows = cur.fetchall()
            if not rows:
                messagebox.showinfo("Query result", "Query returned 0 rows")
                return

            # Show results in a new window
            win = tk.Toplevel(self)
            win.title("Query Results")
            tree = ttk.Treeview(win, columns=list(rows[0].keys()), show='headings')
            tree.pack(fill=tk.BOTH, expand=True)
            for c in rows[0].keys():
                tree.heading(c, text=c)
            for r in rows:
                tree.insert('', tk.END, values=[r[c] for c in r.keys()])

        except Exception as e:
            messagebox.showerror("Query", f"Query failed: {e}")


def main(argv):
    db_path = argv[1] if len(argv) > 1 else None
    app = SQLiteViewer(db_path=db_path)
    app.mainloop()


if __name__ == '__main__':
    main(sys.argv)
