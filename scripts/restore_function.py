    def _generate_with_decoder_and_syntax(self, ctx: CognitiveResponseContext) -> Optional[str]:
        """Generate response using full syntactic pipeline:
        
        1. PrefrontalWorkspace: Plan discourse (3-sentence intent arc)
        2. SyntacticCellAssembly: Bind concepts to grammatical frames
        3. SurfaceRealizer: Apply morphology, agreement, pronouns, articles
        
        Uses graph walk chain concepts directly - decoder produces garbage.
        """
        subject = ctx.subject
        assocs = ctx.associated_concepts
        if not subject or not assocs:
            return None
        
        # ─── STEP 1: PrefrontalWorkspace — Discourse Planning ───
        plan = self.pfc_workspace.plan_discourse(
            user_input=ctx.raw_input,
            subject=subject,
            concept_pos=self._concept_pos,
            associations=assocs[:10],
            past_topics=ctx.past_topics,
            is_follow_up=self._is_follow_up(ctx.raw_input),
        )
        
        # ─── STEP 2: For each discourse intent, generate a sentence ───
        sentences = []
        dopamine_tone = getattr(self, '_dopamine_tone', 0.5)
        used_targets = {subject.lower()}  # Track used concepts across sentences
        
        for sent_idx, intent in enumerate(plan.intents):
            # Generate concept sequence for this intent
            intent_subject = intent.subject
            intent_target = intent.target_concept
            intent_relation = intent.primary_relation
            
            # Filter out grammatical concepts from discourse plan target
            if intent_target and intent_target.lower() in self._GRAMMATICAL_CONCEPTS:
                intent_target = ""
            
            # Walk chain for this intent's relation type
            seen = {intent_subject.lower()}
            if intent_target:
                seen.add(intent_target.lower())
            
            chain_result = self._walk_chain(
                label=intent_subject,
                seen=seen,
                max_hops=2,
                temperature=0.25,
                contradiction_penalty=0.6,
                activation_boost=self._activation_boost,
                subject_proximity=intent_subject,
            )
            
            # Parse chain into concepts
            chain_concepts = []
            chain_connectors = []
            if chain_result:
                for token in chain_result.split():
                    if token in self._CONNECTOR_SET:
                        chain_connectors.append(token)
                    else:
                        chain_concepts.append(token)
            
            # Use discourse target if chain didn't produce it, else use chain's top concept
            target_for_frame = intent_target
            if not target_for_frame and chain_concepts:
                for c in chain_concepts:
                    cl = c.lower()
                    if cl not in seen and cl not in self._GRAMMATICAL_CONCEPTS and cl not in used_targets:
                        target_for_frame = c
                        break
            
            if not target_for_frame:
                neighbor = self._find_vector_neighbor(intent_subject)
                if neighbor and neighbor.lower() not in used_targets and neighbor.lower() not in self._GRAMMATICAL_CONCEPTS:
                    target_for_frame = neighbor
                else:
                    target_for_frame = intent_target
            
            if not target_for_frame or target_for_frame.lower() in used_targets or target_for_frame.lower() in self._GRAMMATICAL_CONCEPTS:
                continue
            
            used_targets.add(target_for_frame.lower())
            
            # ─── STEP 4: SyntacticCellAssembly — Bind to grammatical frame ───
            frame = self.syntactic_assembly.bind_to_sentence(
                subject=intent_subject,
                relation=intent_relation,
                target=target_for_frame,
                pos_map=self._concept_pos,
                chain_concepts=chain_concepts,
                chain_connectors=chain_connectors,
                depth=sent_idx,
            )
            
            # Apply discourse marker from plan
            discourse_marker = intent.discourse_marker
            
            # ─── STEP 5: SurfaceRealizer — Morphology, agreement, pronouns, articles ───
            discourse_state = DiscourseState(
                sentence_index=sent_idx,
                previous_subject=sentences[-1].split()[0].lower() if sentences else None,
                discourse_type=intent.type,
                total_sentences=len(plan.intents),
            )
            
            try:
                sentence = self.surface_realizer.realize(
                    frame=frame,
                    discourse_context=discourse_state,
                    dopamine_tone=dopamine_tone,
                    cerebellar_ngram=getattr(self, 'cerebellar_ngram', None),
                    discourse_marker=discourse_marker,
                )
                sentences.append(sentence)
            except Exception as e:
                if self._trace_enabled:
                    print(f"  [trace] SurfaceRealizer error: {e}")
                continue
        
        if not sentences:
            return None
        
        return " ".join(sentences)

    def _reasoning_loop(self, ctx: CognitiveResponseContext)