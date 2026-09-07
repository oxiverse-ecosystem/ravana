            _toks = [w for w in re.findall(r"[a-z']+", _tail)
                     if w not in ("about", "on", "the", "a", "an", "of", "for",
                                  "with", "to", "we", "should", "could", "would",
                                  "is", "are", "do", "does", "you", "your",
                                  "i", "i'm", "i've", "i'd", "i'll", "my", "me",
                                  "we're", "our", "us", "they", "them", "he",
                                  "she", "his", "her", "its", "their", "it",
                                  "that", "this", "and", "or")]
            # DEFECT A/B/GENERALIZE: take the MAXIMAL noun phrase after the opinion cue.
            _i = 0
            while _i < len(_toks) and (_toks[_i] in ("about", "on", "the", "a", "an",
                    "of", "for", "with", "to", "we", "should", "could", "would",
                    "is", "are", "do", "does", "you", "your", "i", "i'm", "i've",
                    "i'd", "i'll", "my", "me", "we're", "our", "us", "they",
                    "them", "he", "she", "his", "her", "its", "their", "it",
                    "that", "this", "and", "or") or _toks[_i].isdigit()):
                _i += 1
            if _i >= len(_toks):
                _agent_opinion = None
                _stance, _reason = None, None
            else:
                _target = _toks[_i]
                j = _i + 1
                while j < len(_toks) and _toks[j] not in (
                        "and", "or", "that", "this", "the", "a", "an", "of", "for",
                        "with", "to", "is", "are", "do", "does", "you", "your",
                        "my", "me", "it", "its", "they", "them", "he", "she",
                        "his", "her", "their", "we", "our", "us", "i", "i'm",
                        "we're"):
                    _target += " " + _toks[j]
                    j += 1
                _stance, _reason = self._agent_stance_on(_target)
            _reason = (_reason or "").rstrip()
            if _reason and not _reason.endswith((".", "!", "?")):
                _reason += "."
            # Fail-open: when no real topic object was extracted...
            if _stance is None:
                return None
