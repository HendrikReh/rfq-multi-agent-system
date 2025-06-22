#!/usr/bin/env python3
"""
Scenario Viewer

Utility for viewing and analyzing recorded RFQ scenario JSON files.
Provides summaries, detailed views, and basic analytics of scenario runs.
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from agents import ScenarioRecorder


def list_scenarios(reports_dir: str = "./reports") -> List[Dict]:
    """
    List all scenario files with summaries.
    
    Args:
        reports_dir: Directory containing scenario reports
        
    Returns:
        List of scenario summaries
    """
    recorder = ScenarioRecorder(reports_dir)
    scenario_files = recorder.list_scenario_files()
    
    if not scenario_files:
        print(f"No scenario files found in {reports_dir}")
        return []
    
    summaries = []
    for filepath in sorted(scenario_files):
        try:
            summary = recorder.get_scenario_summary(filepath)
            summary['filepath'] = filepath
            summaries.append(summary)
        except Exception as e:
            print(f"❌ Error reading {filepath}: {e}")
    
    return summaries


def display_scenario_list(summaries: List[Dict]):
    """Display a formatted list of scenarios."""
    if not summaries:
        print("No scenarios to display.")
        return
    
    print("📋 RECORDED SCENARIOS")
    print("=" * 80)
    print(f"{'ID':<3} {'Name':<30} {'Timestamp':<20} {'Quote':<8} {'Value':<12} {'Questions':<10}")
    print("-" * 80)
    
    for summary in summaries:
        timestamp = datetime.fromisoformat(summary['timestamp']).strftime("%Y-%m-%d %H:%M")
        quote_status = "✅" if summary['quote_generated'] else "❌"
        quote_value = f"${summary['total_quote_value']:,.0f}" if summary['total_quote_value'] else "N/A"
        
        print(f"{summary['scenario_id']:<3} {summary['scenario_name'][:29]:<30} {timestamp:<20} "
              f"{quote_status:<8} {quote_value:<12} {summary['questions_asked']:<10}")


def display_scenario_details(filepath: str):
    """Display detailed information about a specific scenario."""
    recorder = ScenarioRecorder()
    
    try:
        scenario_data = recorder.load_scenario(filepath)
    except Exception as e:
        print(f"❌ Error loading scenario: {e}")
        return
    
    print("📊 SCENARIO DETAILS")
    print("=" * 80)
    
    # Metadata
    metadata = scenario_data['metadata']
    print(f"Scenario ID: {metadata['scenario_id']}")
    print(f"Name: {metadata['scenario_name']}")
    print(f"Timestamp: {metadata['timestamp']}")
    print(f"File: {metadata['filename']}")
    print()
    
    # Customer Profile
    profile = scenario_data['customer_profile']
    print("👤 CUSTOMER PROFILE")
    print("-" * 40)
    print(f"Persona: {profile['persona']}")
    print(f"Business Context: {profile['business_context']}")
    print()
    
    # Conversation Flow
    flow = scenario_data['conversation_flow']
    print("💬 CONVERSATION FLOW")
    print("-" * 40)
    print(f"Initial Inquiry: \"{flow['initial_inquiry']}\"")
    
    if flow['customer_responses']:
        print("\nCustomer Responses:")
        for i, response in enumerate(flow['customer_responses'], 1):
            print(f"{i}. \"{response[:100]}{'...' if len(response) > 100 else ''}\"")
    
    if flow['quote_response']:
        print(f"\nQuote Response: \"{flow['quote_response'][:100]}{'...' if len(flow['quote_response']) > 100 else ''}\"")
    print()
    
    # Analytics
    analytics = scenario_data['analytics']
    print("📈 ANALYTICS")
    print("-" * 40)
    print(f"Decision Confidence: {analytics['decision_confidence']}/5")
    print(f"Should Ask Questions: {analytics['should_ask_questions']}")
    print(f"Requirements Completeness: {analytics['requirements_completeness']}")
    print(f"Customer Urgency: {analytics['customer_urgency']}/5")
    print(f"Price Sensitivity: {analytics['customer_price_sensitivity']}/5")
    print(f"Buying Readiness: {analytics['customer_readiness_to_buy']}/5")
    print(f"Questions Generated: {analytics['questions_generated']}")
    print(f"Quote Generated: {analytics['quote_generated']}")
    
    if analytics['total_quote_value']:
        print(f"Total Quote Value: ${analytics['total_quote_value']:,.2f}")
        print(f"Quote Line Items: {analytics['quote_line_items']}")
    
    if analytics.get('question_priorities'):
        priorities = analytics['question_priorities']
        print(f"Average Question Priority: {priorities['average_priority']:.1f}")
        print(f"Priority Range: {priorities['min_priority']} - {priorities['max_priority']}")
    
    if analytics.get('performance_metrics'):
        perf = analytics['performance_metrics']
        print(f"\n📊 PERFORMANCE METRICS")
        print(f"Response Time: {perf['response_time']:.2f}s")
        print(f"Accuracy Score: {perf['accuracy_score']:.2f}")
        print(f"Satisfaction Prediction: {perf['customer_satisfaction_prediction']:.2f}")
    
    # Agent Models
    agent_models = scenario_data.get('agent_models', {})
    if agent_models:
        print(f"\n🤖 AGENT MODELS")
        print("-" * 40)
        for agent_type, model in agent_models.items():
            print(f"{agent_type:25} → {model}")
    
    # Error Information
    if scenario_data.get('error_info'):
        error_info = scenario_data['error_info']
        print(f"\n❌ ERROR INFORMATION")
        print(f"Error Type: {error_info['error_type']}")
        print(f"Error Message: {error_info['error_message']}")


def analyze_scenarios(summaries: List[Dict]):
    """Provide analytics across all scenarios."""
    if not summaries:
        print("No scenarios to analyze.")
        return
    
    print("📊 SCENARIO ANALYTICS")
    print("=" * 80)
    
    total_scenarios = len(summaries)
    quote_generated = sum(1 for s in summaries if s['quote_generated'])
    error_scenarios = sum(1 for s in summaries if s['error_occurred'])
    
    print(f"Total Scenarios: {total_scenarios}")
    print(f"Quotes Generated: {quote_generated} ({quote_generated/total_scenarios*100:.1f}%)")
    print(f"Error Scenarios: {error_scenarios} ({error_scenarios/total_scenarios*100:.1f}%)")
    
    # Quote value statistics
    quote_values = [s['total_quote_value'] for s in summaries if s['total_quote_value']]
    if quote_values:
        print(f"\n💰 QUOTE VALUE STATISTICS")
        print(f"Average Quote Value: ${sum(quote_values)/len(quote_values):,.2f}")
        print(f"Min Quote Value: ${min(quote_values):,.2f}")
        print(f"Max Quote Value: ${max(quote_values):,.2f}")
        print(f"Total Quote Volume: ${sum(quote_values):,.2f}")
    
    # Question statistics
    questions_asked = [s['questions_asked'] for s in summaries]
    if questions_asked:
        print(f"\n❓ QUESTION STATISTICS")
        print(f"Average Questions per Scenario: {sum(questions_asked)/len(questions_asked):.1f}")
        print(f"Max Questions in Scenario: {max(questions_asked)}")
        print(f"Scenarios with Questions: {sum(1 for q in questions_asked if q > 0)} ({sum(1 for q in questions_asked if q > 0)/total_scenarios*100:.1f}%)")
    
    # Confidence statistics
    confidence_levels = [s['decision_confidence'] for s in summaries if s['decision_confidence']]
    if confidence_levels:
        print(f"\n🎯 CONFIDENCE STATISTICS")
        print(f"Average Decision Confidence: {sum(confidence_levels)/len(confidence_levels):.1f}/5")
        print(f"High Confidence Scenarios (4-5): {sum(1 for c in confidence_levels if c >= 4)} ({sum(1 for c in confidence_levels if c >= 4)/len(confidence_levels)*100:.1f}%)")


def main():
    """Main function for the scenario viewer."""
    parser = argparse.ArgumentParser(description="View and analyze recorded RFQ scenarios")
    parser.add_argument(
        "--dir", 
        default="./reports",
        help="Directory containing scenario reports (default: ./reports)"
    )
    parser.add_argument(
        "--details",
        help="Show detailed view of specific scenario file"
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Show analytics across all scenarios"
    )
    
    args = parser.parse_args()
    
    if args.details:
        display_scenario_details(args.details)
    else:
        summaries = list_scenarios(args.dir)
        
        if args.analyze:
            analyze_scenarios(summaries)
        else:
            display_scenario_list(summaries)
            
            if summaries:
                print(f"\nUse --details <filepath> to view detailed information")
                print(f"Use --analyze to see analytics across all scenarios")


if __name__ == "__main__":
    main() 