import logging
import json
import re
import regex
import pandas as pd
from ollama import Client
import argparse
from pathlib import Path
from datetime import datetime


class AnonymizationPipeline:
    def __init__(
        self,
        regex_file="regex_rules.json",
        llm_model="qwen2.5-coder:32b",
        continue_from=None,
        text_removal_threshold=15.0,
        results_dir="results_"
    ):
        # Set up logging
        self.logger = logging.getLogger("AnonymizationPipeline")
        self.logger.setLevel(logging.DEBUG)

        # File handler for detailed logging
        fh = logging.FileHandler(
            f"{results_dir}{llm_model.replace(':', '_')}/anonymization.log"
        )
        fh.setLevel(logging.DEBUG)
        fh_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(fh_formatter)

        # Console handler for important info
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch_formatter = logging.Formatter("%(message)s")
        ch.setFormatter(ch_formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        # Store the regex file path and load rules
        self.regex_file = regex_file
        self.regex_rules = self.load_regex_rules()
        self.logger.info(f"Initialized pipeline with {len(self.regex_rules)} regex rules")

        self.ollama_client = Client()
        self.model = llm_model
        self.continue_from = continue_from

        self.processed_reports = []
        self.processed_reports_anonymized = []
        self.current_df = None
        self.current_report_index = None
        self.report_column = None

        # Store threshold
        self.text_removal_threshold = text_removal_threshold
        self.logger.info(f"Initialized pipeline with {text_removal_threshold}% text removal threshold")

    def load_regex_rules(self):
        with open(self.regex_file, "r") as f:
            return json.load(f)

    def save_regex_rules(self):
        with open(self.regex_file, "w") as f:
            json.dump(self.regex_rules, f, indent=4)

    def apply_regex_rules(self, text):

        current_text = text

        # First apply manual rules
        for rule in self.regex_rules:
            if rule.get("valid", True) and "Manual rule" in rule["description"]:
                try:
                    current_text = regex.sub(
                        rule["pattern"],
                        " ",
                        current_text,
                        flags=regex.IGNORECASE,
                        timeout=10
                    )
                except TimeoutError:
                    self.logger.warning(f"Timeout applying rule: {rule}")
                    continue

        original_length = len(current_text)
        modified_text = current_text
        rules_to_invalidate = []

        # Apply only valid rules one by one and check impact
        for rule in self.regex_rules:
            if not rule.get(
                "valid", True # Skip invalid rules, default to True if 'valid' not present
            ) and "Manual rule" in rule["description"]:  # Manual rules applied above
                continue

            text_before = modified_text
            try:
                modified_text = regex.sub(
                    rule["pattern"], " ", 
                    modified_text, 
                    flags=regex.IGNORECASE,
                    timeout=10
                )
            except TimeoutError:
                rules_to_invalidate.append(rule)
                continue

            # Check if this specific rule removed too much text
            chars_removed = len(text_before) - len(modified_text)
            removal_percentage = (chars_removed / len(text_before))

            if removal_percentage > self.text_removal_threshold and rule.get("description", "").startswith("LLM-generated"):
                self.logger.warning(
                    f"Rule '{rule['pattern']}' removed {removal_percentage*100:.1f}% of text - marking as invalid"
                )
                rules_to_invalidate.append(rule)

        # Update validity of problematic rules
        if rules_to_invalidate:
            for rule in rules_to_invalidate:
                rule["valid"] = False
                self.logger.info(f"Marked rule as invalid: {rule['pattern']}")
            # Save updated rules
            self.save_regex_rules()
            # Reapply only valid rules
            modified_text = text
            for rule in self.regex_rules:
                if rule.get("valid", True):
                    modified_text = regex.sub(
                        rule["pattern"],
                        " ",
                        modified_text,
                        flags=regex.IGNORECASE,
                        timeout=5
                    )

        # Final check for overall text removal
        if len(modified_text) < original_length * (1 - self.text_removal_threshold/100):
            self.logger.error(
                f"Total text removal too high: {((original_length - len(modified_text)) / original_length) * 100:.1f}%"
            )
            raise ValueError("Warning: Excessive text removal detected!")

        return modified_text

    def chunk_text(self, text, chunk_size=5000, overlap=50):
        """Split text into overlapping chunks for very long reports"""
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + chunk_size

            # If this is not the last chunk, try to break at a sentence
            if end < text_length:
                # Look for sentence endings in the overlap region
                overlap_start = max(end - overlap, start)
                overlap_text = text[overlap_start:end]

                # Try to find sentence endings (., !, ?)
                sentence_ends = [
                    i + overlap_start
                    for i, char in enumerate(overlap_text)
                    if char in ".!?"
                    and (i + 1 >= len(overlap_text) or overlap_text[i + 1].isspace())
                ]

                if sentence_ends:
                    # Use the last sentence ending in the overlap region
                    end = sentence_ends[-1] + 1

            chunk = text[start:end]
            chunks.append(chunk)

            # Start next chunk from the last sentence end
            start = end if end < text_length else text_length

        self.logger.debug(f"Split text into {len(chunks)} chunks")
        return chunks

    def detect_phi_with_llm(self, text):
        # Split text into chunks if it's too long
        chunks = self.chunk_text(text) if len(text) > 5000 else [text]
        
        for i, chunk in enumerate(chunks, 1):
            self.logger.debug(f"Processing chunk {i}/{len(chunks)} ({len(chunk)} chars)")
            
            prompt = (
                f"You are working as a PHI detection system, with the aim to anonymize radiology reports. "
                f"PHI according to HIPAA is anything that may be used to identify the patient. It includes:\n"
                f"- Patient names\n"
                f"- Medical record numbers (MRN)\n"
                f"- Accession numbers\n"
                f"- All elements of dates (except year)\n"
                f"- Phone numbers\n"
                f"- Addresses and all geographical subdivisions smaller than a State\n"
                f"- Email addresses\n"
                f"- Any other identifiers such as social security numbers, license or serial numbers etc.\n"
                f"Mere size measurements, examination type, clinical history, and other non-identifying information are not PHI.\n"
                f"If you find PHI, provide ONE regex pattern that captures ONE type of PHI found.\n"
                f"For most cases, the regex pattern should capture the underlying format, not just the specific PHI instance.\n"
                f"For example, instead of capturing 'MRN: 123456' as PHI, capture 'MRN:' and the following six digits.\n"
                f"However, do not make overly greedy patterns that would match too much text.\n"
                f"If you do not find any PHI, respond with 'NO_PHI_PRESENT'.\n"
                f"Do NOT summarize the report.\n" 
                f"Do NOT create patterns that match placeholders or already masked/redacted content.\n"
                f"Respond ONLY in valid JSON format with exactly these two elements:\n"
                f'{{"Reasoning": "explain which specific PHI element you found", "regex": "either NO_PHI_PRESENT or a regex pattern for that specific PHI"}}'
                f"Here is a first example of a valid response:\n"
                f'{{"Reasoning": "Found the following phone number (323)-232 32 21 - potential PHI.", "regex": "\\(\\d{3}\\)-\\d{3} \\d{2} \\d{2}"}}\n'
                f'Here is a second example of a valid response:\n'
                f'{{"Reasoning": "Found patient name: John Doe - which is PHI.", "regex": "John Doe"}}\n'
                f"Here is a third example of a valid response:\n"
                f'{{"Reasoning": "Found string MRN but no number - already anonymized or placeholder, so nothing to do in this instance.", "regex": "NO_PHI_PRESENT"}}\n'
                f"Report to analyze:\n{chunk}"
            )

            try:
                response = self.ollama_client.generate(
                    model=self.model,
                    prompt=prompt,
                    format="json",
                    options={"temperature": 0.7}
                )
                llm_response = response["response"]
                self.logger.debug(f"Raw LLM Response for chunk {i}: {llm_response}")
                
                # Clean and parse response
                # Clean and parse response
                if "```json" in llm_response:
                    llm_response = llm_response.split("```json")[1].split("```")[0].strip()
                elif "```" in llm_response:
                    llm_response = llm_response.split("```")[1].strip()

                # Add brackets if they're missing
                if not llm_response.strip().startswith('{'):
                    llm_response = '{' + llm_response + '}'

                try:
                    parsed_response = json.loads(llm_response)
                    
                    # If we found PHI in this chunk, return it immediately
                    if parsed_response["regex"] != "NO_PHI_PRESENT":
                        self.logger.info(f"Found PHI in chunk {i}")
                        return parsed_response
                        
                except json.JSONDecodeError as e:
                    self.logger.error(f"Failed to parse JSON in chunk {i}: {e}")
                    continue
                    
            except Exception as e:
                self.logger.error(f"Error processing chunk {i}: {str(e)}")
                continue

        # If we get here, no PHI was found in any chunk
        return {"Reasoning": "No PHI found in any chunk", "regex": "NO_PHI_PRESENT"}

    def validate_regex(self, pattern, text):
        """Validate that the regex pattern is valid and not too broad"""

        try:
            # Easy check if pattern would match too much text
            if pattern.endswith(r'[A-Za-z]+') or pattern.endswith(r'[A-Z][a-z]+'): # frequent in llama models
                self.logger.warning(f"Pattern '{pattern}' would match too much text - rejecting")
                return False, "Pattern matches too much text"
            
            regex.compile(pattern)
            # Test the pattern on the text
            matches = regex.findall(
                pattern, 
                text, 
                flags=regex.IGNORECASE,
                timeout=10
            )

            # Log what the pattern would match
            self.logger.debug(f"Pattern '{pattern}' found matches: {matches}")

            # Handle both string matches and tuple matches (from capture groups)
            if not matches:
                return False, "Pattern matches no text"
            elif matches and isinstance(matches[0], tuple):
                # If matches are tuples, flatten them and join
                match_text = "".join("".join(t) for t in matches)
            else:
                # If matches are strings, just join them
                match_text = "".join(matches)

            # Check if pattern matches are reasonable (not too broad)
            match_length = len(match_text)
            text_length = len(text)
            match_percentage = (match_length / text_length) * 100

            if match_length > text_length * 0.3:
                self.logger.warning(
                    f"Pattern '{pattern}' matches {match_percentage:.1f}% of text - too broad"
                )
                return False, "Pattern matches too much text"

            self.logger.debug(
                f"Pattern '{pattern}' matches {match_percentage:.1f}% of text - acceptable"
            )
            return True, None

        except regex.error as e:
            self.logger.warning(f"Invalid regex pattern '{pattern}': {str(e)}")
            return False, f"Invalid regex pattern: {str(e)}"

    def validate_rule_against_future(self, new_pattern, future_reports):
        """
        Validate a new regex rule against future reports.
        Ensures no report is altered by more than the threshold percentage.
        """
        for idx, report in enumerate(future_reports):
            try:
                # Apply all existing valid rules to get current state
                current_text = report
                for rule in self.regex_rules:
                    if rule.get("valid", True):
                        current_text = regex.sub(
                            rule["pattern"],
                            " ",
                            current_text,
                            flags=regex.IGNORECASE,
                            timeout=5
                        )

                # Apply the new pattern
                test_text = regex.sub(
                    new_pattern,
                    " ",
                    current_text,
                    flags=regex.IGNORECASE,
                    timeout=5
                )

                # Calculate percentage of text removed
                chars_removed = len(current_text) - len(test_text)
                removal_percentage = (chars_removed / len(current_text)) * 100

                if removal_percentage > self.text_removal_threshold:
                    self.logger.warning(
                        f"Rule '{new_pattern}' alters future report {idx + 1} by {removal_percentage:.1f}% - rejecting"
                    )
                    return False

            except TimeoutError:
                self.logger.warning(f"Timeout while testing rule on future report {idx + 1}")
                return False

        self.logger.info(f"Rule '{new_pattern}' is valid for future reports")
        return True

    def process_report(self, report):
        self.logger.info("\n--- Processing new report ---")
        self.logger.debug(f"Original report: {report[:500]}...")

        current_text = report
        iteration = 0
        max_iterations = 25
        min_iterations = 3

        while iteration < max_iterations:
            iteration += 1
            self.logger.info(f"\nIteration {iteration}/{max_iterations}")

            # First apply existing rules
            try:
                current_text = self.apply_regex_rules(current_text)
                self.logger.debug(f"After applying existing rules: {current_text[:500]}...")
            except ValueError as e:
                self.logger.error(f"Error applying existing rules: {str(e)}")
                raise

            # Check with LLM for remaining PHI
            try:
                llm_response = self.detect_phi_with_llm(current_text)

                if llm_response["regex"] == "NO_PHI_PRESENT":
                    if iteration >= min_iterations:
                        self.logger.info(f"No PHI found after {iteration} iterations")
                        break
                    else:
                        self.logger.info(f"No PHI found but continuing to minimum {min_iterations} iterations")
                        continue

                # Extract and validate new regex pattern
                new_pattern = llm_response["regex"].strip()
                phi_type = llm_response["Reasoning"]
                self.logger.info(f"Found PHI: {phi_type}")
                self.logger.info(f"Suggested pattern: {new_pattern}")

                # Validate the new regex
                is_valid, error_msg = self.validate_regex(new_pattern, current_text)
                if not is_valid:
                    self.logger.error(f"Regex validation failed: {error_msg}")
                    continue  # Skip this pattern but continue iterations

                # Only validate against history if we have processed enough reports
                if len(self.processed_reports) >= 20:
                    is_safe = self.validate_rule_against_history(new_pattern)
                    if not is_safe:
                        self.logger.warning(f"Rule rejected: affects too many previous reports")
                        continue
                else:
                    self.logger.info(
                        f"Only {len(self.processed_reports)} reports processed so far - "
                        f"skipping historical validation for rule '{new_pattern}'"
                    )

                # Validate against future reports
                future_reports = self.get_future_reports()
                is_safe_for_future = self.validate_rule_against_future(new_pattern, future_reports)
                if not is_safe_for_future:
                    self.logger.warning(f"Rule rejected: affects future reports too much")
                    continue

                # Add new rule
                new_rule = {
                    "description": f"LLM-generated rule for {phi_type} (iter {iteration})",
                    "pattern": new_pattern,
                    "flags": "IGNORECASE",
                    "valid": True,
                }
                self.regex_rules.append(new_rule)
                self.save_regex_rules()
                self.logger.info(f"Added new rule: {new_rule}")

                # Apply the new rule
                current_text = regex.sub(
                    new_pattern, 
                    " ", 
                    current_text, 
                    flags=regex.IGNORECASE,
                    timeout=10
                )

            except Exception as e:
                self.logger.error(f"Error in iteration {iteration}: {str(e)}")
                continue  # Try next iteration despite error

        if iteration >= max_iterations:
            self.logger.warning(f"Reached maximum iterations ({max_iterations})")

        # Store the processed report for future validation
        self.processed_reports.append(report)
        self.processed_reports_anonymized.append(current_text)

        return current_text

    def validate_rule_against_history(self, new_pattern):
        """
        Validate a new regex rule against previously processed reports using batch processing.
        """
        affected_reports = 0
        total_reports = len(self.processed_reports)
        batch_size = min(50, total_reports)
        
        for i in range(0, total_reports, batch_size):
            batch = self.processed_reports[i:i + batch_size]
            
            for idx, original_report in enumerate(batch, start=i):
                try:
                    # First apply all existing valid rules to get current state
                    current_text = original_report
                    for rule in self.regex_rules:
                        if rule.get("valid", True):
                            current_text = regex.sub(
                                rule["pattern"],
                                " ",
                                current_text,
                                flags=regex.IGNORECASE,
                                timeout=5
                            )
                    
                    # Now apply the new pattern and check if it causes additional changes
                    test_text = regex.sub(
                        new_pattern,
                        " ",
                        current_text,
                        flags=regex.IGNORECASE,
                        timeout=5
                    )
                    
                    # Check if the new rule caused additional changes
                    if test_text != current_text:
                        affected_reports += 1
                        self.logger.debug(f"Rule would affect previously processed report {idx + 1}")
                        
                        # Early exit if we've already exceeded our threshold
                        if affected_reports / total_reports > self.text_removal_threshold/100:
                            self.logger.warning(
                                f"Rule '{new_pattern}' affects {affected_reports} out of {total_reports} "
                                f"previous reports ({(affected_reports/total_reports)*100:.1f}%) - rejecting"
                            )
                            return False
                            
                except TimeoutError:
                    self.logger.warning(f"Timeout while testing rule on report {idx + 1}")
                    return False
                    
        self.logger.info(
            f"Rule '{new_pattern}' affects {affected_reports} out of {total_reports} "
            f"previous reports ({(affected_reports/total_reports)*100:.1f}%) - accepted"
        )
        return True

    @staticmethod
    def get_last_processed_report(log_file="anonymization.log"):
        """Parse the log file to find the last successfully processed report number"""
        if not Path(log_file).exists():
            return 0

        last_report = 0
        processing_pattern = re.compile(r"Processing report (\d+)/\d+")

        with open(log_file, "r") as f:
            for line in f:
                if match := processing_pattern.search(line):
                    current_report = int(match.group(1))
                    last_report = max(last_report, current_report)

        return last_report

    def process_excel_file(self, excel_path, report_column, max_reports=None):
        # Store DataFrame and column name for future reference
        self.current_df = pd.read_excel(excel_path)
        self.report_column = report_column
        
        # Read Excel file and remove duplicates
        df = pd.read_excel(excel_path)
        initial_count = len(df)
        df = df.drop_duplicates(subset=[report_column], keep="first")
        duplicates_removed = initial_count - len(df)

        if max_reports:
            df = df.iloc[:max_reports]

        total_reports = len(df)

        self.logger.info(f"\nRemoved {duplicates_removed} duplicate reports")

        # Determine starting point
        start_idx = 0
        if self.continue_from:
            start_idx = self.continue_from - 1  # Convert to 0-based index
            self.logger.info(f"Continuing from report {self.continue_from}")

        self.logger.info(
            f"Processing {total_reports - start_idx} reports from {excel_path}"
        )

        # Initialize results list with existing results if continuing
        anonymized_reports = []
        if start_idx > 0:
            # Fill with placeholders for already processed reports
            anonymized_reports = ["PREVIOUSLY_PROCESSED"] * start_idx

        # Process reports
        for idx, report in enumerate(df[report_column].iloc[start_idx:], start_idx + 1):
            try:
                self.current_report_index = idx - 1  # Store current position (0-based index)
                self.logger.info(f"\nProcessing report {idx}/{total_reports}")
                anonymized_report = self.process_report(report)
                anonymized_reports.append(anonymized_report)
                self.logger.info(f"Successfully anonymized report {idx}")
            except ValueError as e:
                error_msg = f"Error processing report {idx}: {str(e)}"
                self.logger.error(error_msg)
                anonymized_reports.append("ERROR: " + str(e))

        df[f"{report_column}_anonymized"] = anonymized_reports
        success_count = sum(
            1
            for x in anonymized_reports
            if not str(x).startswith("ERROR") and x != "PREVIOUSLY_PROCESSED"
        )
        self.logger.info(
            f"\nProcessing complete: {success_count}/{total_reports - start_idx} new reports successfully anonymized"
        )

        # Log summary statistics
        self.logger.info("\nSummary:")
        self.logger.info(f"Initial reports: {initial_count}")
        self.logger.info(f"Duplicate reports removed: {duplicates_removed}")
        self.logger.info(f"Unique reports processed: {total_reports}")

        return df

    def get_rules_status(self):
        """Get statistics about current rules"""
        total_rules = len(self.regex_rules)
        valid_rules = sum(1 for rule in self.regex_rules if rule.get("valid", True))
        invalid_rules = total_rules - valid_rules

        self.logger.info("\nRules Status:")
        self.logger.info(f"Total rules: {total_rules}")
        self.logger.info(f"Valid rules: {valid_rules}")
        self.logger.info(f"Invalid rules: {invalid_rules}")

        if invalid_rules > 0:
            self.logger.info("\nInvalid rules:")
            for rule in self.regex_rules:
                if not rule.get("valid", True):
                    self.logger.info(f"- {rule['description']}: {rule['pattern']}")

    def get_future_reports(self, look_ahead=200):
        """
        Retrieve a sample of future reports for validation.
        
        Args:
            look_ahead (int): Number of future reports to retrieve
            
        Returns:
            list: List of future reports, empty list if none available
        """
        if self.current_df is None or self.current_report_index is None:
            self.logger.debug("No DataFrame available or current position unknown")
            return []

        total_reports = len(self.current_df)
        start_idx = self.current_report_index + 1  # Start from next report
        
        if start_idx >= total_reports:
            self.logger.debug("No future reports available - at end of dataset")
            return []
            
        end_idx = min(start_idx + look_ahead, total_reports)
        
        future_reports = self.current_df[self.report_column].iloc[start_idx:end_idx].tolist()
        self.logger.debug(f"Retrieved {len(future_reports)} future reports for validation")
        
        return future_reports


def main():
    parser = argparse.ArgumentParser(
        description="Anonymize medical reports in Excel file"
    )
    parser.add_argument(
        "excel_file",
        default="radiology-full-text.xlsx",
        nargs="?",
        help="Path to Excel file containing reports",
    )
    parser.add_argument(
        "report_column",
        default="narrative",
        nargs="?",
        help="Name of column containing reports",
    )
    parser.add_argument(
        "-c",
        "--continue",
        action="store_true",
        help="Continue from last processed report in log file",
    )
    args = parser.parse_args()

    continue_from = None
    if getattr(args, "continue"):
        continue_from = AnonymizationPipeline.get_last_processed_report(
            "anonymization.log"
        )
        if continue_from == 0:
            print("No previous progress found in log file. Starting from beginning.")
        else:
            print(f"Continuing from report {continue_from + 1}")

    pipeline = AnonymizationPipeline(continue_from=continue_from)

    # Create output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_path = Path(args.excel_file)
    output_path = (
        input_path.parent
        / f"{input_path.stem}_anonymized_{timestamp}{input_path.suffix}"
    )

    # Process file and save results
    df = pipeline.process_excel_file(args.excel_file, args.report_column)
    df.to_excel(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
