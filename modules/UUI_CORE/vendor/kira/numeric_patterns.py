"""Dimensionless numeric change learner. No graph nodes, edges or spatial assumptions.

A pattern is an observed exact rational delta (next - previous) conditioned on
an exact rational previous change. This is deliberately narrow: no claim that
numeric proximity alone implies semantic similarity or causality.
"""
import sqlite3
from fractions import Fraction
from numeric_core import rational

class NumericPatternLearner:
    def __init__(self, path=':memory:'):
        self.db=sqlite3.connect(path)
        self.db.execute('CREATE TABLE IF NOT EXISTS numeric_patterns (context_num TEXT NOT NULL, context_den TEXT NOT NULL, delta_num TEXT NOT NULL, delta_den TEXT NOT NULL, hits INTEGER NOT NULL, PRIMARY KEY(context_num,context_den,delta_num,delta_den))')
        self.db.execute('CREATE TABLE IF NOT EXISTS numeric_series (series TEXT PRIMARY KEY, last_num TEXT NOT NULL, last_den TEXT NOT NULL, prev_delta_num TEXT, prev_delta_den TEXT)')
        self.db.commit()
    def observe(self, series, value):
        v=rational(value)
        with self.db:
            row=self.db.execute('SELECT last_num,last_den,prev_delta_num,prev_delta_den FROM numeric_series WHERE series=?',(series,)).fetchone()
            if row is None:
                self.db.execute('INSERT INTO numeric_series VALUES (?,?,?,?,?)',(series,str(v.numerator),str(v.denominator),None,None))
                return None
            last=Fraction(int(row[0]),int(row[1])); delta=v-last
            if row[2] is not None:
                context=Fraction(int(row[2]),int(row[3]))
                self.db.execute('INSERT INTO numeric_patterns VALUES (?,?,?,?,1) ON CONFLICT(context_num,context_den,delta_num,delta_den) DO UPDATE SET hits=hits+1',(str(context.numerator),str(context.denominator),str(delta.numerator),str(delta.denominator)))
            self.db.execute('UPDATE numeric_series SET last_num=?,last_den=?,prev_delta_num=?,prev_delta_den=? WHERE series=?',(str(v.numerator),str(v.denominator),str(delta.numerator),str(delta.denominator),series))
            return delta
    def predict(self, series):
        row=self.db.execute('SELECT last_num,last_den,prev_delta_num,prev_delta_den FROM numeric_series WHERE series=?',(series,)).fetchone()
        if row is None or row[2] is None: return None
        last=Fraction(int(row[0]),int(row[1])); context=Fraction(int(row[2]),int(row[3]))
        options=self.db.execute('SELECT delta_num,delta_den,hits FROM numeric_patterns WHERE context_num=? AND context_den=? ORDER BY hits DESC, delta_num,delta_den',(str(context.numerator),str(context.denominator))).fetchall()
        if not options: return None
        best=options[0]; delta=Fraction(int(best[0]),int(best[1]))
        return {'value':last+delta,'delta':delta,'support':best[2],'alternatives':len(options)}
    def patterns(self):
        return self.db.execute('SELECT COUNT(*) FROM numeric_patterns').fetchone()[0]
    def close(self): self.db.close()
